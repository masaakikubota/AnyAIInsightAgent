from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .models import (
    PersonaBuildJobConfig,
    PersonaBuildJobProgress,
    PersonaBuildJobResponse,
    PersonaBuildJobStatus,
)
from .services.google_sheets import (
    GoogleSheetsError,
    batch_update_values,
    extract_spreadsheet_id,
    find_sheet,
)
from .services.persona_builder import generate_persona_from_blueprint


@dataclass
class PersonaBuildJob:
    job_id: str
    config: PersonaBuildJobConfig
    status: PersonaBuildJobStatus = PersonaBuildJobStatus.pending
    stage: str = "pending"
    total_blueprints: int = 0
    generated_personas: int = 0
    message: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0
    task: Optional[asyncio.Task] = None


class PersonaBuildJobManager:
    def __init__(self, base_dir: Path) -> None:
        self.root_dir = base_dir
        self.base_dir = base_dir / "persona_build"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, PersonaBuildJob] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, job_id: str, cfg: PersonaBuildJobConfig) -> PersonaBuildJobResponse:
        async with self._lock:
            job = PersonaBuildJob(job_id=job_id, config=cfg)
            self.jobs[job_id] = job
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
        job.task = asyncio.create_task(self._run_job(job))
        return PersonaBuildJobResponse(job_id=job_id, status=job.status)

    def get_progress(self, job_id: str) -> PersonaBuildJobProgress:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return PersonaBuildJobProgress(
            job_id=job.job_id,
            status=job.status,
            stage=job.stage,
            total_blueprints=job.total_blueprints,
            generated_personas=job.generated_personas,
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

    async def _run_job(self, job: PersonaBuildJob) -> None:
        job.status = PersonaBuildJobStatus.running
        job.stage = "load_blueprints"
        job.started_at = time.time()
        job.message = "ペルソナブループリントを読み込んでいます…"
        job_dir = self._job_dir(job.job_id)
        output_dir = self._resolve_output_dir(job.config.output_dir, job_dir)

        try:
            blueprint_path = self._resolve_path(job.config.blueprint_path)
            if not blueprint_path.exists():
                raise FileNotFoundError(f"blueprint_path not found: {blueprint_path}")

            blueprints = json.loads(blueprint_path.read_text(encoding="utf-8"))
            if not isinstance(blueprints, list):
                raise ValueError("ブループリントファイルはJSON配列である必要があります")

            if job.config.persona_goal and job.config.persona_goal > 0:
                blueprints = blueprints[: job.config.persona_goal]

            job.total_blueprints = len(blueprints)
            if not blueprints:
                raise ValueError("ブループリントが存在しません。Seed Ingest v0.2 の出力を確認してください。")

            persona_results: List[dict] = []
            usage_logs: List[dict] = []
            semaphore = asyncio.Semaphore(job.config.concurrency)
            progress_lock = asyncio.Lock()

            job.stage = "persona_generation"
            job.message = f"{job.total_blueprints} 件のブループリントからペルソナを生成しています…"

            async def _produce(idx: int, blueprint: dict) -> None:
                nonlocal persona_results
                async with semaphore:
                    persona, usage = await generate_persona_from_blueprint(
                        job.config,
                        blueprint,
                        persona_index=idx,
                        seed_offset=job.config.persona_seed_offset,
                    )
                async with progress_lock:
                    persona_results.append(persona)
                    job.generated_personas = len(persona_results)
                    usage_logs.append(
                        {
                            "blueprint_id": blueprint.get("blueprint_id"),
                            "persona_id": persona.get("persona_id"),
                            "usage": usage,
                        }
                    )
                    job.message = (
                        f"ペルソナ生成中 {job.generated_personas}/{job.total_blueprints} "
                        f"(最新: {persona.get('persona_id')})"
                    )

            tasks = [asyncio.create_task(_produce(idx, blueprint)) for idx, blueprint in enumerate(blueprints)]
            await asyncio.gather(*tasks)

            persona_results.sort(key=lambda item: item.get("persona_id", ""))

            catalog_path = output_dir / "persona_catalog.json"
            catalog_path.parent.mkdir(parents=True, exist_ok=True)
            catalog_path.write_text(json.dumps(persona_results, ensure_ascii=False, indent=2), encoding="utf-8")

            jsonl_path = output_dir / "persona_catalog.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as fout:
                for persona in persona_results:
                    fout.write(json.dumps(persona, ensure_ascii=False) + "\n")

            usage_path = output_dir / "persona_generation_usage.json"
            usage_path.write_text(json.dumps(usage_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            job.artifacts["persona_catalog"] = str(catalog_path.relative_to(self.root_dir))
            job.artifacts["persona_catalog_jsonl"] = str(jsonl_path.relative_to(self.root_dir))
            job.artifacts["persona_generation_usage"] = str(usage_path.relative_to(self.root_dir))

            report = await self._build_ingest_report(job.config, blueprint_path, persona_results, usage_logs)
            report_path = output_dir / "persona_generation_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            job.artifacts["persona_generation_report"] = str(report_path.relative_to(self.root_dir))

            if job.config.persona_sheet_url:
                job.stage = "sheet_export"
                job.message = "生成したペルソナをGoogle Sheetsへ書き込み中…"
                await self._write_personas_to_sheet(job, persona_results)

            job.message = f"{job.generated_personas} 件のペルソナ生成が完了しました。"
            job.stage = "completed"
            job.status = PersonaBuildJobStatus.completed
        except Exception as exc:  # noqa: BLE001
            job.status = PersonaBuildJobStatus.failed
            job.stage = "failed"
            job.message = f"ペルソナ生成に失敗しました: {exc}"
        finally:
            job.finished_at = time.time()
            summary = {
                "job_id": job.job_id,
                "status": job.status.value,
                "stage": job.stage,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "total_blueprints": job.total_blueprints,
                "generated_personas": job.generated_personas,
                "artifacts": job.artifacts,
                "message": job.message,
            }
            (self._job_dir(job.job_id) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            job.artifacts.setdefault("summary", str((self._job_dir(job.job_id) / "summary.json").relative_to(self.root_dir)))

    def _resolve_output_dir(self, configured: Optional[str], job_dir: Path) -> Path:
        if configured:
            path = self._resolve_path(configured)
            path.mkdir(parents=True, exist_ok=True)
            return path
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    async def _build_ingest_report(
        self,
        cfg: PersonaBuildJobConfig,
        blueprint_path: Path,
        personas: List[dict],
        usage_logs: List[dict],
    ) -> dict:
        try:
            blueprint_relative = blueprint_path.relative_to(self.root_dir)
            blueprint_path_repr = str(blueprint_relative)
        except ValueError:
            blueprint_path_repr = str(blueprint_path)

        blueprint_meta = {
            "source_blueprint_path": blueprint_path_repr,
            "persona_goal": cfg.persona_goal,
            "generated_personas": len(personas),
        }

        region_counter: Dict[str, int] = {}
        attitude_counter: Dict[str, int] = {}
        for persona in personas:
            region = (persona.get("region") or "unspecified").strip() or "unspecified"
            region_counter[region] = region_counter.get(region, 0) + 1
            attitude = persona.get("attitude_cluster") or "unspecified"
            attitude_counter[attitude] = attitude_counter.get(attitude, 0) + 1

        return {
            "project_name": cfg.project_name,
            "domain": cfg.domain,
            "language": cfg.language,
            "blueprint": blueprint_meta,
            "region_distribution": region_counter,
            "attitude_distribution": attitude_counter,
            "usage_logs": usage_logs,
        }

    async def _write_personas_to_sheet(self, job: PersonaBuildJob, personas: List[dict]) -> None:
        cfg = job.config
        sheet_url = cfg.persona_sheet_url
        if not sheet_url:
            return

        try:
            spreadsheet_id = extract_spreadsheet_id(sheet_url)
        except GoogleSheetsError as exc:
            job.message = f"シートIDの解析に失敗しました: {exc}"
            return

        sheet_name = cfg.persona_sheet_name or "PersonaCatalog"
        try:
            match = await asyncio.to_thread(find_sheet, spreadsheet_id, sheet_name)
            sheet_name = match.sheet_name
        except GoogleSheetsError:
            sheet_name = sheet_name or "PersonaCatalog"

        start_row = cfg.persona_start_row
        overview_column = cfg.persona_overview_column.strip().upper() or "B"
        prompt_column = cfg.persona_prompt_column.strip().upper() or "C"

        overview_values: List[List[str]] = []
        prompt_values: List[List[str]] = []
        for persona in personas:
            overview = (
                f"{persona.get('age_band','-')} / {persona.get('income_band','-')} / "
                f"{persona.get('region','-')} | Attitude: {persona.get('attitude_cluster','-')}"
            )
            motivations = persona.get("motivations") or []
            frictions = persona.get("frictions") or []
            summary = persona.get("summary") or ""
            prompt_lines = [summary]
            if motivations:
                prompt_lines.append("Motivations: " + "; ".join(motivations))
            if frictions:
                prompt_lines.append("Frictions: " + "; ".join(frictions))
            if persona.get("quote"):
                prompt_lines.append(f"Quote: \"{persona['quote']}\"")
            prompt_values.append(["\n".join(prompt_lines)])
            overview_values.append([overview])

        def _build_range(column_letter: str, values: List[List[str]]) -> Optional[dict]:
            if not values:
                return None
            col_letter = column_letter.strip().upper() or "B"
            if not col_letter.isalpha():
                col_letter = "B"
            end_row = start_row + len(values) - 1
            rng = f"'{sheet_name}'!{col_letter}{start_row}:{col_letter}{end_row}"
            return {"range": rng, "values": values}

        data = list(filter(None, [_build_range(overview_column, overview_values), _build_range(prompt_column, prompt_values)]))
        if not data:
            return

        try:
            await asyncio.to_thread(batch_update_values, spreadsheet_id, data)
            job.artifacts["persona_sheet"] = (
                f"{sheet_name}!{overview_column}{start_row}:{prompt_column}{start_row + len(personas) - 1}"
            )
        except GoogleSheetsError as exc:
            job.message = f"Google Sheetsへの書き込みに失敗しました: {exc}"
