from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import polars as pl

from .models import (
    DashboardFilters,
    DashboardFilterOptions,
    DashboardQueryResponse,
    DashboardRunDetail,
    DashboardRunSummary,
    DashboardRequest,
    PersonaResponseJobConfig,
    PersonaResponseJobProgress,
    PersonaResponseJobResponse,
    PersonaResponseJobStatus,
)
from .services.google_sheets import (
    GoogleSheetsError,
    extract_spreadsheet_id,
    fetch_sheet_values,
    find_sheet,
)
from .services.persona_responder import generate_response_for_persona
from .services.ssr_mapper import ReferenceConfig, SSRMappingError, map_responses_to_pmfs
from .services.dashboard import build_dashboard_file
from .services.dashboard_data import (
    build_dashboard_from_base,
    build_persona_insight_request,
    prepare_dashboard_base,
)
from .services.clients import call_openai_dashboard_plan


@dataclass
class PersonaResponseJob:
    job_id: str
    config: PersonaResponseJobConfig
    status: PersonaResponseJobStatus = PersonaResponseJobStatus.pending
    stage: str = "pending"
    total_pairs: int = 0
    processed_pairs: int = 0
    message: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0
    task: Optional[asyncio.Task] = None


class PersonaResponseJobManager:
    def __init__(self, base_dir: Path) -> None:
        self.root_dir = base_dir
        self.base_dir = base_dir / "persona_responses"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, PersonaResponseJob] = {}
        self._lock = asyncio.Lock()

    CACHE_TTL_SECONDS = 60 * 60 * 24

    async def create_job(self, job_id: str, cfg: PersonaResponseJobConfig) -> PersonaResponseJobResponse:
        async with self._lock:
            job = PersonaResponseJob(job_id=job_id, config=cfg)
            self.jobs[job_id] = job
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
        job.task = asyncio.create_task(self._run_job(job))
        return PersonaResponseJobResponse(job_id=job_id, status=job.status)

    def get_progress(self, job_id: str) -> PersonaResponseJobProgress:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return PersonaResponseJobProgress(
            job_id=job.job_id,
            status=job.status,
            stage=job.stage,
            total_pairs=job.total_pairs,
            processed_pairs=job.processed_pairs,
            message=job.message,
            artifacts=dict(job.artifacts) if job.artifacts else None,
        )

    def _job_dir(self, job_id: str) -> Path:
        return self.base_dir / job_id

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = (self.root_dir / path).resolve()
        return path

    def _output_dir_from_config(self, configured: Optional[str], job_dir: Path) -> Path:
        if configured:
            return self._resolve_path(configured)
        return job_dir

    def _resolve_output_dir(self, configured: Optional[str], job_dir: Path) -> Path:
        path = self._output_dir_from_config(configured, job_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _column_letter_to_index(self, column: str) -> int:
        col = (column or "A").strip().upper()
        if not col.isalpha():
            raise ValueError(f"Invalid column reference: {column}")
        index = 0
        for char in col:
            index = index * 26 + (ord(char) - ord("A") + 1)
        return index - 1

    def _relative_to_root(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root_dir))
        except ValueError:
            return str(path)

    async def _run_job(self, job: PersonaResponseJob) -> None:
        job.status = PersonaResponseJobStatus.running
        job.stage = "load_resources"
        job.started_at = time.time()
        job.message = "ペルソナカタログと刺激を読み込んでいます…"
        job_dir = self._job_dir(job.job_id)
        output_dir = self._resolve_output_dir(job.config.output_dir, job_dir)

        try:
            personas = self._load_personas(job.config)
            stimuli = await self._load_stimuli(job.config)

            if not personas:
                raise ValueError("ペルソナカタログが空です。")
            if not stimuli:
                raise ValueError("刺激が見つかりません。")

            persona_limit = job.config.persona_limit or len(personas)
            stimuli_limit = job.config.stimuli_limit or len(stimuli)
            personas = personas[:persona_limit]
            stimuli = stimuli[:stimuli_limit]

            pairs: List[tuple[int, dict, str]] = []
            for persona_idx, persona in enumerate(personas):
                for stimulus_idx, stimulus in enumerate(stimuli):
                    pairs.append((len(pairs), persona, stimulus))

            job.total_pairs = len(pairs)
            if not pairs:
                raise ValueError("生成対象のペルソナ×刺激組み合わせがありません。")

            semaphore = asyncio.Semaphore(job.config.concurrency)
            progress_lock = asyncio.Lock()
            responses: List[dict] = []
            usage_logs: List[dict] = []

            job.stage = "response_generation"
            job.message = f"{job.total_pairs}件の組み合わせに対して応答を生成しています…"

            async def _produce(pair_index: int, persona: dict, stimulus: str) -> None:
                async with semaphore:
                    response, usage = await generate_response_for_persona(
                        job.config,
                        persona,
                        stimulus,
                        pair_index=pair_index,
                    )
                record = {
                    "persona_id": persona.get("persona_id"),
                    "persona_summary": persona.get("summary"),
                    "stimulus": stimulus,
                    "response": response.get("persona_response"),
                    "structured": response.get("structured"),
                }
                async with progress_lock:
                    responses.append(record)
                    job.processed_pairs = len(responses)
                    usage_logs.append(
                        {
                            "persona_id": record["persona_id"],
                            "stimulus": stimulus,
                            "usage": usage,
                        }
                    )
                    job.message = (
                        f"応答生成中 {job.processed_pairs}/{job.total_pairs} "
                        f"(最新: {record['persona_id']} x {stimulus[:24]})"
                    )

            tasks = [asyncio.create_task(_produce(idx, persona, stimulus)) for idx, persona, stimulus in pairs]
            await asyncio.gather(*tasks)

            responses.sort(key=lambda item: (item.get("persona_id") or "", item.get("stimulus") or ""))

            responses_path = output_dir / "persona_responses.json"
            responses_path.write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding="utf-8")
            job.artifacts["persona_responses"] = str(responses_path.relative_to(self.root_dir))

            jsonl_path = output_dir / "persona_responses.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as fout:
                for record in responses:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            job.artifacts["persona_responses_jsonl"] = str(jsonl_path.relative_to(self.root_dir))

            usage_path = output_dir / "persona_response_usage.json"
            usage_path.write_text(json.dumps(usage_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            job.artifacts["persona_response_usage"] = str(usage_path.relative_to(self.root_dir))

            ssr_path: Optional[Path] = None
            if job.config.ssr_reference_path and job.config.ssr_reference_set:
                job.stage = "ssr_mapping"
                job.message = "SSR マッピングを実行しています…"
                ssr_path = await self._generate_ssr(job, responses, output_dir)
                job.artifacts["persona_responses_ssr"] = str(ssr_path.relative_to(self.root_dir))

            report_path = output_dir / "persona_response_report.json"
            report = self._build_report(job.config, personas, stimuli, responses)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            job.artifacts["persona_response_report"] = str(report_path.relative_to(self.root_dir))

            await self._generate_dashboard_artifacts(job, personas, stimuli, responses, ssr_path, output_dir)

            job.message = f"{job.processed_pairs} 件の応答生成が完了しました。"
            job.stage = "completed"
            job.status = PersonaResponseJobStatus.completed
        except Exception as exc:  # noqa: BLE001
            job.status = PersonaResponseJobStatus.failed
            job.stage = "failed"
            job.message = f"ペルソナ応答生成に失敗しました: {exc}"
        finally:
            job.finished_at = time.time()
            summary = {
                "job_id": job.job_id,
                "status": job.status.value,
                "stage": job.stage,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "total_pairs": job.total_pairs,
                "processed_pairs": job.processed_pairs,
                "artifacts": job.artifacts,
                "message": job.message,
            }
            (self._job_dir(job.job_id) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            job.artifacts.setdefault("summary", str((self._job_dir(job.job_id) / "summary.json").relative_to(self.root_dir)))

    def _load_personas(self, cfg: PersonaResponseJobConfig) -> List[dict]:
        path = self._resolve_path(cfg.persona_catalog_path)
        if not path.exists():
            raise FileNotFoundError(f"persona_catalog_path not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        raise ValueError("persona_catalog_path は JSON 配列である必要があります。")

    async def _load_stimuli(self, cfg: PersonaResponseJobConfig) -> List[str]:
        manual = self._parse_manual_stimuli(cfg.stimuli_source)
        if cfg.stimuli_sheet_url:
            try:
                sheet_values = await self._read_stimuli_from_sheet(cfg)
                if sheet_values:
                    return sheet_values
            except Exception:
                pass
        return manual or ["Baseline Concept"]

    def _parse_manual_stimuli(self, text: Optional[str]) -> List[str]:
        if not text:
            return []
        items = []
        for line in text.replace("\r\n", "\n").split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            parts = [part.strip() for part in stripped.split(",") if part.strip()]
            if parts:
                items.extend(parts)
        return items

    async def _read_stimuli_from_sheet(self, cfg: PersonaResponseJobConfig) -> List[str]:
        spreadsheet_id = extract_spreadsheet_id(cfg.stimuli_sheet_url or "")
        sheet_name = cfg.stimuli_sheet_name
        if not sheet_name:
            try:
                match = await asyncio.to_thread(find_sheet, spreadsheet_id, "ProductStimuli")
                sheet_name = match.sheet_name
            except GoogleSheetsError:
                sheet_name = "Sheet1"
        rows = await asyncio.to_thread(fetch_sheet_values, spreadsheet_id, sheet_name)
        if not rows:
            return []
        column_index = self._column_letter_to_index(cfg.stimuli_sheet_column)
        start_row = max(0, cfg.stimuli_sheet_start_row - 1)
        values: List[str] = []
        for idx, row in enumerate(rows):
            if idx < start_row:
                continue
            cell = row[column_index] if column_index < len(row) else ""
            value = str(cell).strip()
            if value:
                values.append(value)
        return values

    async def _generate_ssr(
        self,
        job: PersonaResponseJob,
        responses: List[dict],
        output_dir: Path,
    ) -> Path:
        cfg = job.config
        ref = ReferenceConfig(
            reference_path=Path(cfg.ssr_reference_path).expanduser(),
            embeddings_column=cfg.ssr_embeddings_column,
            model_name=cfg.ssr_model_name,
            device=cfg.ssr_device,
        )
        texts: List[str] = []
        entries: List[dict] = []
        for record in responses:
            response_text = (record.get("response") or "").strip()
            if not response_text:
                continue
            entries.append(
                {
                    "persona_id": record.get("persona_id"),
                    "stimulus": record.get("stimulus"),
                    "response": response_text,
                }
            )
            texts.append(response_text)
        if not texts:
            raise ValueError("SSRを実行するための応答テキストがありません。")
        try:
            pmfs = await asyncio.to_thread(
                map_responses_to_pmfs,
                texts,
                reference_set=cfg.ssr_reference_set,
                config=ref,
                temperature=cfg.ssr_temperature,
                epsilon=cfg.ssr_epsilon,
            )
        except SSRMappingError as exc:
            raise RuntimeError(f"SSR変換に失敗しました: {exc}") from exc

        for entry, pmf in zip(entries, pmfs):
            entry["pmf"] = pmf
        out_path = output_dir / "persona_responses_ssr.json"
        out_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def _build_report(
        self,
        cfg: PersonaResponseJobConfig,
        personas: List[dict],
        stimuli: List[str],
        responses: List[dict],
    ) -> dict:
        per_persona: Dict[str, int] = {}
        per_stimulus: Dict[str, int] = {}
        for record in responses:
            pid = record.get("persona_id") or "unknown"
            stim = record.get("stimulus") or "unknown"
            per_persona[pid] = per_persona.get(pid, 0) + 1
            per_stimulus[stim] = per_stimulus.get(stim, 0) + 1
        return {
            "project_name": cfg.project_name,
            "domain": cfg.domain,
            "language": cfg.language,
            "persona_count": len(personas),
            "stimuli_count": len(stimuli),
            "total_pairs": len(responses),
            "responses_per_persona": per_persona,
            "responses_per_stimulus": per_stimulus,
        }

    def list_dashboard_runs(self) -> List[DashboardRunSummary]:
        if not self.base_dir.exists():
            return []
        summaries: List[DashboardRunSummary] = []
        for job_dir in sorted(self.base_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            summary = self._load_dashboard_summary(job_dir)
            if summary:
                summaries.append(summary)
        summaries.sort(
            key=lambda item: (item.updated_at or item.created_at or datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )
        return summaries

    def get_dashboard_run(self, job_id: str) -> DashboardRunDetail:
        job_dir = self._job_dir(job_id)
        if not job_dir.exists():
            raise KeyError(job_id)
        summary = self._load_dashboard_summary(job_dir)
        if not summary:
            raise KeyError(job_id)
        config_path = job_dir / "config.json"
        cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
        cfg = PersonaResponseJobConfig.model_validate(cfg_data)
        return DashboardRunDetail(**summary.model_dump(), config=cfg)

    def query_dashboard(
        self,
        job_id: str,
        filters: Optional[DashboardFilters],
        limit: Optional[int],
        include_records: bool = True,
    ) -> DashboardQueryResponse:
        ctx = self._load_dashboard_context(job_id)
        filters_obj = filters or DashboardFilters()
        limit_val = int(limit) if isinstance(limit, int) and limit > 0 else None
        include_records_flag = bool(include_records)

        base_df, expected_stimuli, signature, base_status = self._get_base_dataframe(job_id, ctx)
        filters_dict = self._filters_to_dict(filters_obj)
        cache_key = self._filters_cache_key(filters_dict, limit_val, include_records_flag)
        query_status, cached_payload = self._load_query_cache(
            ctx,
            cache_key,
            signature,
            limit_val,
            include_records_flag,
        )

        if cached_payload:
            cached_response = DashboardQueryResponse.model_validate(cached_payload)
            cached_response.cache_status = "hit"
            return cached_response

        request, filtered_df, meta = build_dashboard_from_base(
            base_df,
            expected_stimuli,
            filters=filters_obj,
        )

        available_filtered = meta.get("filter_options_filtered")
        available_all = meta.get("filter_options_all")

        records: Optional[List[Dict[str, str | float | None]]] = None
        if include_records_flag:
            records = self._convert_records(filtered_df, limit_val)

        cache_status = "miss"
        if base_status == "miss":
            cache_status = "rebuild"

        response = DashboardQueryResponse(
            request=request,
            total_responses=int(meta.get("total_responses", 0)),
            filtered_responses=int(meta.get("filtered_responses", 0)),
            total_personas=int(meta.get("total_personas", 0)),
            filtered_personas=int(meta.get("filtered_personas", 0)),
            total_stimuli=int(meta.get("total_stimuli", 0)),
            filtered_stimuli=int(meta.get("filtered_stimuli", 0)),
            filters=filters_obj,
            available_filters=available_filtered or DashboardFilterOptions(),
            available_filters_all=available_all or DashboardFilterOptions(),
            records=records,
            cache_status=cache_status,
        )

        self._store_query_cache(
            ctx,
            cache_key,
            signature,
            limit_val,
            include_records_flag,
            filters_dict,
            response,
        )

        return response

    def _load_dashboard_summary(self, job_dir: Path) -> Optional[DashboardRunSummary]:
        config_path = job_dir / "config.json"
        if not config_path.exists():
            return None
        cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
        cfg = PersonaResponseJobConfig.model_validate(cfg_data)

        summary_path = job_dir / "summary.json"
        summary_data: Dict[str, object] = {}
        if summary_path.exists():
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))

        output_dir = self._output_dir_from_config(cfg.output_dir, job_dir)
        created_at_ts = summary_data.get("started_at") if isinstance(summary_data, dict) else None
        finished_at_ts = summary_data.get("finished_at") if isinstance(summary_data, dict) else None
        created_at = (
            datetime.fromtimestamp(float(created_at_ts), tz=timezone.utc)
            if isinstance(created_at_ts, (int, float))
            else datetime.fromtimestamp(job_dir.stat().st_ctime, tz=timezone.utc)
        )
        updated_at = (
            datetime.fromtimestamp(float(finished_at_ts), tz=timezone.utc)
            if isinstance(finished_at_ts, (int, float))
            else datetime.fromtimestamp(job_dir.stat().st_mtime, tz=timezone.utc)
        )

        artifacts = {}
        if isinstance(summary_data, dict):
            artifacts = summary_data.get("artifacts") or {}
            if not isinstance(artifacts, dict):
                artifacts = {}

        return DashboardRunSummary(
            job_id=job_dir.name,
            project_name=cfg.project_name,
            domain=cfg.domain,
            language=cfg.language,
            output_dir=self._relative_to_root(output_dir),
            created_at=created_at,
            updated_at=updated_at,
            total_pairs=int(summary_data.get("total_pairs", 0)) if isinstance(summary_data, dict) else None,
            artifacts={key: str(value) for key, value in artifacts.items()},
        )

    def _load_dashboard_context(self, job_id: str) -> Dict[str, object]:
        job_dir = self._job_dir(job_id)
        if not job_dir.exists():
            raise FileNotFoundError(job_id)

        config_path = job_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found for job {job_id}")
        cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
        cfg = PersonaResponseJobConfig.model_validate(cfg_data)

        output_dir = self._output_dir_from_config(cfg.output_dir, job_dir)
        responses_path = output_dir / "persona_responses.json"
        if not responses_path.exists():
            raise FileNotFoundError(f"persona_responses.json not found for job {job_id}")
        responses = json.loads(responses_path.read_text(encoding="utf-8"))

        ssr_entries = None
        ssr_path = output_dir / "persona_responses_ssr.json"
        if ssr_path.exists():
            ssr_entries = json.loads(ssr_path.read_text(encoding="utf-8"))

        try:
            personas = self._load_personas(cfg)
        except Exception:
            personas = []

        stimuli = sorted({record.get("stimulus") for record in responses if record.get("stimulus")})
        if not stimuli:
            stimuli = ["Baseline Concept"]

        return {
            "config": cfg,
            "output_dir": output_dir,
            "responses": responses,
            "personas": personas,
            "stimuli": stimuli,
            "ssr_entries": ssr_entries,
            "responses_path": responses_path,
            "ssr_path": ssr_path,
            "cache_dir": output_dir / "cache",
        }

    def _dashboard_cache_dir(self, ctx: Dict[str, object]) -> Path:
        cache_dir = ctx["cache_dir"]
        if not isinstance(cache_dir, Path):
            cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _source_signature(self, ctx: Dict[str, object]) -> str:
        responses_path: Path = ctx["responses_path"]
        parts = [
            str(responses_path.stat().st_size),
            str(responses_path.stat().st_mtime_ns),
            str(len(ctx.get("responses") or [])),
        ]
        ssr_path: Optional[Path] = ctx.get("ssr_path")
        if ssr_path and ssr_path.exists():
            parts.append(str(ssr_path.stat().st_size))
            parts.append(str(ssr_path.stat().st_mtime_ns))
        payload = "|".join(parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_base_dataframe(
        self,
        job_id: str,
        ctx: Dict[str, object],
    ) -> Tuple[pl.DataFrame, int, str, str]:
        cache_dir = self._dashboard_cache_dir(ctx)
        base_path = cache_dir / "base.parquet"
        meta_path = cache_dir / "base_meta.json"
        signature = self._source_signature(ctx)
        now = time.time()

        if base_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
            if (
                meta.get("source_signature") == signature
                and now - float(meta.get("generated_at", 0)) <= self.CACHE_TTL_SECONDS
            ):
                try:
                    base_df = pl.read_parquet(base_path)
                    expected = int(meta.get("expected_stimuli", 0) or 0)
                    if expected <= 0:
                        expected = len(set(ctx["stimuli"]))
                    return base_df, expected, signature, "hit"
                except Exception:
                    pass

        base_df, expected = prepare_dashboard_base(
            ctx["personas"],
            ctx["responses"],
            ctx["stimuli"],
            ctx["ssr_entries"],
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        base_df.write_parquet(base_path)
        meta = {
            "job_id": job_id,
            "generated_at": now,
            "source_signature": signature,
            "expected_stimuli": expected,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return base_df, expected, signature, "miss"

    def _filters_to_dict(self, filters: DashboardFilters) -> Dict[str, object]:
        data = filters.model_dump(exclude_none=True, exclude_defaults=True)
        segments = data.get("segments") or {}
        normalized_segments: Dict[str, List[str]] = {}
        for key, values in segments.items():
            if values:
                normalized_segments[key] = sorted(set(values))
        if normalized_segments:
            data["segments"] = dict(sorted(normalized_segments.items()))
        elif "segments" in data:
            data.pop("segments", None)
        if "stimuli" in data and not data["stimuli"]:
            data.pop("stimuli", None)
        if "persona_ids" in data and not data["persona_ids"]:
            data.pop("persona_ids", None)
        return data

    def _filters_cache_key(
        self,
        filters_dict: Dict[str, object],
        limit: Optional[int],
        include_records: bool,
    ) -> str:
        payload = {
            "filters": filters_dict,
            "limit": limit or 0,
            "include_records": include_records,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _load_query_cache(
        self,
        ctx: Dict[str, object],
        cache_key: str,
        signature: str,
        limit: Optional[int],
        include_records: bool,
    ) -> Tuple[str, Optional[dict]]:
        cache_dir = self._dashboard_cache_dir(ctx)
        cache_file = cache_dir / f"query_{cache_key}.json"
        if not cache_file.exists():
            return "miss", None
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "stale", None
        if payload.get("source_signature") != signature:
            return "stale", None
        if payload.get("limit") != (limit or 0):
            return "stale", None
        if bool(payload.get("include_records", True)) != include_records:
            return "stale", None
        response_payload = payload.get("response")
        if not response_payload:
            return "stale", None
        generated_at = float(payload.get("generated_at", 0))
        if time.time() - generated_at > self.CACHE_TTL_SECONDS:
            return "stale", None
        return "hit", response_payload

    def _store_query_cache(
        self,
        ctx: Dict[str, object],
        cache_key: str,
        signature: str,
        limit: Optional[int],
        include_records: bool,
        filters_dict: Dict[str, object],
        response: DashboardQueryResponse,
    ) -> None:
        cache_dir = self._dashboard_cache_dir(ctx)
        cache_file = cache_dir / f"query_{cache_key}.json"
        cacheable = response.model_copy(update={"cache_status": "hit"})
        payload = {
            "generated_at": time.time(),
            "source_signature": signature,
            "limit": limit or 0,
            "include_records": include_records,
            "filters": filters_dict,
            "response": cacheable.model_dump(mode="json"),
        }
        cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _convert_records(self, df, limit: Optional[int]) -> List[Dict[str, str | float | None]]:
        if limit is not None and limit > 0:
            df = df.head(limit)
        rows = df.to_dicts()
        cleaned: List[Dict[str, str | float | None]] = []
        for row in rows:
            normalized: Dict[str, str | float | None] = {}
            for key, value in row.items():
                if isinstance(value, float) and math.isnan(value):
                    normalized[key] = None
                else:
                    normalized[key] = value
            cleaned.append(normalized)
        return cleaned

    async def _generate_dashboard_artifacts(
        self,
        job: PersonaResponseJob,
        personas: List[dict],
        stimuli: List[str],
        responses: List[dict],
        ssr_path: Optional[Path],
        output_dir: Path,
    ) -> None:
        ssr_entries = None
        if ssr_path and ssr_path.exists():
            ssr_entries = json.loads(ssr_path.read_text(encoding="utf-8"))

        request, aggregated_df, _meta = build_persona_insight_request(personas, responses, stimuli, ssr_entries)
        request_path = output_dir / "dashboard_request.json"
        request_path.write_text(request.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
        csv_path = output_dir / "persona_responses_aggregated.csv"
        aggregated_df.write_csv(csv_path)

        html_path = output_dir / "persona_insight_dashboard.html"
        plan_path = output_dir / "persona_insight_plan.md"
        await build_dashboard_file(request, html_path, plan_path)

        summary_path = output_dir / "persona_insight_summary.md"
        summary_text = await _generate_dashboard_summary(request)
        summary_path.write_text(summary_text, encoding="utf-8")

        job.artifacts["persona_dashboard_request"] = str(request_path.relative_to(self.root_dir))
        job.artifacts["persona_dashboard_csv"] = str(csv_path.relative_to(self.root_dir))
        job.artifacts["persona_dashboard_html"] = str(html_path.relative_to(self.root_dir))
        job.artifacts["persona_dashboard_plan"] = str(plan_path.relative_to(self.root_dir))
        job.artifacts["persona_dashboard_summary"] = str(summary_path.relative_to(self.root_dir))


async def _generate_dashboard_summary(request: DashboardRequest) -> str:
    system_prompt = (
        "You are a senior insights analyst. Summarize dashboard datasets into key findings and actions. "
        "Return markdown with sections: Highlights, Notable Segments, Next Actions."
    )
    user_prompt = request.model_dump_json(indent=2, ensure_ascii=False)
    summary, _ = await call_openai_dashboard_plan(system_prompt, user_prompt, model="gpt-5-high", temperature=0.4)
    return summary
