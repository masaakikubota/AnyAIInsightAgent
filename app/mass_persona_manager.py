from __future__ import annotations

import asyncio
import json
import math
import re
import time
import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .models import (
    MassPersonaJobConfig,
    MassPersonaJobProgress,
    MassPersonaJobResponse,
    MassPersonaJobStatus,
    PersonaDirectionConfig,
)
from .services.google_sheets import (
    GoogleSheetsError,
    extract_spreadsheet_id,
    fetch_sheet_values,
    find_sheet,
)
from .services.interview_llm import LLMGenerationError
from .services.persona_direction import generate_direction_matrix


DEFAULT_AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
DEFAULT_INCOME_BANDS = ["low", "mid", "high"]
DEFAULT_REGIONS = ["JP_Kanto", "JP_Kansai", "US_West", "US_East", "EU_West"]
DEFAULT_ATTITUDES = [
    "value_seeker",
    "premium_oriented",
    "eco_conscious",
    "trend_follower",
    "sensitive_skin",
]


@dataclass
class MassPersonaJob:
    job_id: str
    config: MassPersonaJobConfig
    status: MassPersonaJobStatus = MassPersonaJobStatus.pending
    stage: str = "pending"
    total_seed_records: int = 0
    processed_records: int = 0
    generated_blueprints: int = 0
    message: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0
    task: Optional[asyncio.Task] = None


class MassPersonaJobManager:
    def __init__(self, base_dir: Path) -> None:
        self.root_dir = base_dir
        self.base_dir = base_dir / "mass_persona"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, MassPersonaJob] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, job_id: str, cfg: MassPersonaJobConfig) -> MassPersonaJobResponse:
        async with self._lock:
            job = MassPersonaJob(job_id=job_id, config=cfg)
            self.jobs[job_id] = job
        job_dir = self._job_dir(job_id)
        (job_dir / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
        job.task = asyncio.create_task(self._run_job(job))
        return MassPersonaJobResponse(job_id=job_id, status=job.status)

    def get_progress(self, job_id: str) -> MassPersonaJobProgress:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return MassPersonaJobProgress(
            job_id=job.job_id,
            status=job.status,
            stage=job.stage,
            total_seed_records=job.total_seed_records,
            processed_records=job.processed_records,
            generated_blueprints=job.generated_blueprints,
            message=job.message,
            artifacts=dict(job.artifacts) if job.artifacts else None,
        )

    def _job_dir(self, job_id: str) -> Path:
        path = self.base_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def _run_job(self, job: MassPersonaJob) -> None:
        job.status = MassPersonaJobStatus.running
        job.stage = "ingest"
        job.started_at = time.time()
        job.message = "ペルソナ種データを収集中…"
        job_dir = self._job_dir(job.job_id)

        try:
            seed_records = await self._collect_seed_records(job)
            if not seed_records:
                raise ValueError("入力データが見つかりません。手入力またはシート設定を確認してください。")

            job.total_seed_records = len(seed_records)
            job.processed_records = len(seed_records)

            records_path = job_dir / "seed_records.json"
            records_path.write_text(json.dumps(seed_records, ensure_ascii=False, indent=2), encoding="utf-8")
            job.artifacts["seed_records"] = str(records_path.relative_to(self.base_dir))

            _, assignments = await self._ensure_direction_assets(job)

            job.stage = "blueprint"
            job.message = "ペルソナブループリントを生成しています…"
            blueprints, allocation_summary = self._generate_persona_blueprints(job.config, seed_records, assignments)
            job.generated_blueprints = len(blueprints)

            blueprint_path = job_dir / "persona_blueprints.json"
            blueprint_path.write_text(json.dumps(blueprints, ensure_ascii=False, indent=2), encoding="utf-8")
            job.artifacts["persona_blueprints"] = str(blueprint_path.relative_to(self.base_dir))

            report = self._build_ingest_report(job.config, seed_records, allocation_summary)
            report_path = job_dir / "ingest_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            job.artifacts["ingest_report"] = str(report_path.relative_to(self.base_dir))

            job.message = (
                f"{job.total_seed_records}件の発話データから{job.generated_blueprints}件のペルソナブループリントを生成しました。"
            )
            job.stage = "completed"
            job.status = MassPersonaJobStatus.completed
        except Exception as exc:  # noqa: BLE001
            job.status = MassPersonaJobStatus.failed
            job.stage = "failed"
            job.message = f"大量ペルソナ用入力の収集に失敗しました: {exc}"
        finally:
            job.finished_at = time.time()
            summary = {
                "job_id": job.job_id,
                "status": job.status.value,
                "stage": job.stage,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "total_seed_records": job.total_seed_records,
                "processed_records": job.processed_records,
                "generated_blueprints": job.generated_blueprints,
                "artifacts": job.artifacts,
                "message": job.message,
            }
            (job_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            job.artifacts.setdefault("summary", str((job_dir / "summary.json").relative_to(self.base_dir)))

    async def _collect_seed_records(self, job: MassPersonaJob) -> List[dict]:
        cfg = job.config
        manual_records = self._parse_manual_entries(cfg.utterance_source or "", cfg.default_region)
        sheet_records: List[dict] = []

        if cfg.sheet_url:
            try:
                sheet_records = await self._load_sheet_entries(cfg, job)
            except Exception as exc:  # noqa: BLE001
                job.message = f"シート読み込みに失敗しました: {exc}"

        combined = manual_records + sheet_records
        max_records = cfg.max_records or len(combined)
        if max_records and max_records > 0:
            combined = combined[:max_records]
        return combined

    def _parse_manual_entries(self, text: str, default_region: Optional[str]) -> List[dict]:
        if not text:
            return []
        records: List[dict] = []
        for line_no, raw_line in enumerate(text.replace("\r\n", "\n").split("\n"), start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            region: Optional[str] = None
            utterance = stripped
            if "|" in stripped:
                prefix, rest = stripped.split("|", 1)
                if rest.strip():
                    region_candidate = prefix.strip()
                    utterance_candidate = rest.strip()
                    if utterance_candidate:
                        region = region_candidate or default_region
                        utterance = utterance_candidate
            if not utterance:
                continue
            record = {
                "source": "manual",
                "index": line_no,
                "utterance": utterance,
                "region": region or default_region,
            }
            records.append(record)
        return records

    async def _load_sheet_entries(self, cfg: MassPersonaJobConfig, job: MassPersonaJob) -> List[dict]:
        spreadsheet_id = extract_spreadsheet_id(cfg.sheet_url or "")
        sheet_name = cfg.sheet_name
        if not sheet_name:
            try:
                match = await asyncio.to_thread(find_sheet, spreadsheet_id, "PersonaSeeds")
                sheet_name = match.sheet_name
            except GoogleSheetsError:
                sheet_name = "Sheet1"

        rows = await asyncio.to_thread(fetch_sheet_values, spreadsheet_id, sheet_name)
        if not rows:
            return []

        utterance_col = self._column_letter_to_index(cfg.sheet_utterance_column)
        region_col = self._column_letter_to_index(cfg.sheet_region_column) if cfg.sheet_region_column else None
        tags_col = self._column_letter_to_index(cfg.sheet_tags_column) if cfg.sheet_tags_column else None
        start_row = max(0, cfg.sheet_start_row - 1)

        records: List[dict] = []
        for idx, row in enumerate(rows):
            if idx < start_row:
                continue
            utterance_cell = row[utterance_col] if utterance_col < len(row) else ""
            utterance = str(utterance_cell).strip()
            if not utterance:
                continue
            region_val = None
            if region_col is not None and region_col < len(row):
                region_val = str(row[region_col]).strip() or None
            tags_list: Optional[List[str]] = None
            if tags_col is not None and tags_col < len(row):
                raw_tags = str(row[tags_col]).strip()
                if raw_tags:
                    tags_list = [tag.strip() for tag in raw_tags.replace("、", ",").split(",") if tag.strip()]
            record = {
                "source": "sheet",
                "sheet_name": sheet_name,
                "row": idx + 1,
                "utterance": utterance,
                "region": region_val or cfg.default_region,
            }
            if tags_list:
                record["tags"] = tags_list
            records.append(record)
        job.message = f"シートから {len(records)} 件の発話データを読み込みました。"
        return records

    async def _ensure_direction_assets(self, job: MassPersonaJob) -> tuple[str, Dict[str, dict]]:
        cfg = job.config
        axes_config = cfg.direction_axes or {}
        default_axes = {
            "age_band": axes_config.get("age_band") or DEFAULT_AGE_BANDS,
            "income_band": axes_config.get("income_band") or DEFAULT_INCOME_BANDS,
            "region": axes_config.get("region") or DEFAULT_REGIONS,
            "attitude_cluster": axes_config.get("attitude_cluster") or DEFAULT_ATTITUDES,
        }
        persona_goal = cfg.persona_goal
        direction_cfg = PersonaDirectionConfig(
            project_name=cfg.project_name,
            domain=cfg.domain,
            language=cfg.language,
            persona_goal=persona_goal,
            axes=default_axes,
            must_cover_attributes=cfg.direction_must_cover,
            seed_insights=cfg.direction_seed_notes,
            notes=cfg.notes,
        )

        try:
            direction_yaml, assignments = await generate_direction_matrix(direction_cfg)
        except LLMGenerationError as exc:
            job.message = f"方向性LLM生成に失敗しました（{exc}）。フォールバックを使用します。"
            direction_yaml, assignments = self._fallback_direction(cfg, default_axes)
        job_dir = self._job_dir(job.job_id)
        direction_path = job_dir / "persona_direction.yaml"
        direction_path.write_text(direction_yaml, encoding="utf-8")
        job.artifacts["persona_direction"] = str(direction_path.relative_to(self.base_dir))

        assignment_path = job_dir / "persona_assignment.json"
        assignment_path.write_text(json.dumps(assignments, ensure_ascii=False, indent=2), encoding="utf-8")
        job.artifacts["persona_assignment"] = str(assignment_path.relative_to(self.base_dir))
        return direction_yaml, assignments

    def _generate_persona_blueprints(
        self,
        cfg: MassPersonaJobConfig,
        seed_records: List[dict],
        assignments: Dict[str, dict],
    ) -> tuple[List[dict], dict]:
        if not seed_records:
            return [], {
                "target_persona_goal": cfg.persona_goal,
                "effective_blueprints": 0,
                "region_allocation": {},
            }

        normalized_records: List[dict] = []
        region_groups: Dict[str, List[tuple[int, dict]]] = defaultdict(list)
        for idx, record in enumerate(seed_records):
            normalized = dict(record)
            region = (normalized.get("region") or cfg.default_region or "").strip() or "unspecified"
            normalized["region"] = region
            normalized_records.append(normalized)
            region_groups[region].append((idx, normalized))

        ordered_pairs: List[tuple[int, dict]] = []
        for region, pairs in sorted(region_groups.items(), key=lambda item: len(item[1]), reverse=True):
            ordered_pairs.extend(pairs)

        total = len(ordered_pairs)
        target = min(cfg.persona_goal, total) if total else 0
        if target <= 0:
            target = total
        if target <= 0:
            return [], {
                "target_persona_goal": cfg.persona_goal,
                "effective_blueprints": 0,
                "region_allocation": {},
            }

        chunk_size = max(1, math.ceil(total / target))
        blueprints: List[dict] = []
        region_allocation = Counter()

        assignment_items = list(assignments.items())
        for bp_index in range(target):
            start = bp_index * chunk_size
            subset_pairs = ordered_pairs[start : start + chunk_size]
            if not subset_pairs:
                break
            subset_records = [pair[1] for pair in subset_pairs]
            assignment = assignment_items[bp_index % len(assignment_items)] if assignment_items else None
            blueprint = self._build_blueprint(cfg, bp_index, subset_records, assignment)
            blueprints.append(blueprint)
            region_allocation[blueprint["region"]] += 1

        if not blueprints and normalized_records:
            assignment = assignment_items[0] if assignment_items else None
            blueprint = self._build_blueprint(cfg, 0, [normalized_records[0]], assignment)
            blueprints.append(blueprint)
            region_allocation[blueprint["region"]] += 1

        allocation_summary = {
            "target_persona_goal": cfg.persona_goal,
            "effective_blueprints": len(blueprints),
            "region_allocation": dict(region_allocation),
        }
        return blueprints, allocation_summary

    def _build_blueprint(
        self,
        cfg: MassPersonaJobConfig,
        index: int,
        subset_records: List[dict],
        assignment: Optional[tuple[str, dict]],
    ) -> dict:
        blueprint_id = f"bp_{index + 1:04d}"
        region = self._dominant_region(subset_records, cfg)

        sample_utterances = [rec.get("utterance", "") for rec in subset_records[:3] if rec.get("utterance")]
        tag_counter = Counter()
        for rec in subset_records:
            for tag in rec.get("tags") or []:
                tag_counter[tag] += 1
        top_tags = [tag for tag, _ in tag_counter.most_common(5)]

        keywords = self._extract_keywords(subset_records, cfg.language)
        seed_refs = []
        for rec in subset_records:
            ref = {
                "source": rec.get("source"),
                "index": rec.get("index"),
                "row": rec.get("row"),
                "sheet_name": rec.get("sheet_name"),
                "utterance": rec.get("utterance", "")[:160],
            }
            seed_refs.append(ref)

        blueprint = {
            "blueprint_id": blueprint_id,
            "region": region,
            "seed_count": len(subset_records),
            "seed_refs": seed_refs,
            "sample_utterances": sample_utterances,
            "focus_tags": top_tags,
            "focus_keywords": keywords[:8],
            "notes": cfg.notes,
        }
        if assignment and assignment[1].get("attributes"):
            blueprint["assigned_attributes"] = assignment[1]["attributes"]
        if assignment and assignment[1].get("focus"):
            blueprint["persona_theme"] = assignment[1]["focus"]
        elif keywords:
            blueprint["persona_theme"] = self._compose_theme(region, keywords)
        return blueprint

    def _compose_theme(self, region: str, keywords: List[str]) -> str:
        top_keywords = ", ".join(keywords[:3])
        return f"{region} × {top_keywords} 志向" if top_keywords else region

    def _dominant_region(self, records: List[dict], cfg: MassPersonaJobConfig) -> str:
        region_counter = Counter(
            (rec.get("region") or cfg.default_region or "unspecified").strip() or "unspecified"
            for rec in records
        )
        region, _ = region_counter.most_common(1)[0]
        return region

    def _extract_keywords(self, records: List[dict], language: str) -> List[str]:
        tokens: List[str] = []
        for rec in records:
            tokens.extend(rec.get("tags") or [])
            tokens.extend(self._tokenize(rec.get("utterance", ""), language))
        counter = Counter(token for token in tokens if token)
        return [token for token, _ in counter.most_common(12)]

    def _tokenize(self, text: str, language: str) -> List[str]:
        if not text:
            return []
        if language == "ja":
            pattern = r"[ぁ-んァ-ン一-龠A-Za-z0-9]{2,}"
        else:
            pattern = r"[A-Za-z0-9]{2,}"
        return [token.lower() for token in re.findall(pattern, text)]

    def _build_ingest_report(
        self,
        cfg: MassPersonaJobConfig,
        records: List[dict],
        allocation_summary: Optional[dict] = None,
    ) -> dict:
        sources: Dict[str, int] = {}
        regions: Dict[str, int] = {}
        tag_counter = Counter()
        for record in records:
            src = record.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            region = (record.get("region") or cfg.default_region or "unspecified").strip() or "unspecified"
            regions[region] = regions.get(region, 0) + 1
            for tag in record.get("tags") or []:
                tag_counter[tag] += 1

        report = {
            "project_name": cfg.project_name,
            "domain": cfg.domain,
            "language": cfg.language,
            "persona_goal": cfg.persona_goal,
            "max_records": cfg.max_records,
            "source_counts": sources,
            "region_distribution": regions,
        }
        if tag_counter:
            report["tag_distribution"] = dict(tag_counter.most_common(20))
        if allocation_summary:
            report["blueprint_summary"] = allocation_summary
        return report

    def _fallback_direction(self, cfg: MassPersonaJobConfig, axes: Dict[str, List[str]]) -> tuple[str, Dict[str, dict]]:
        direction_yaml = self._build_direction_yaml(cfg, axes)
        age_vals = axes.get("age_band") or DEFAULT_AGE_BANDS
        income_vals = axes.get("income_band") or DEFAULT_INCOME_BANDS
        region_vals = axes.get("region") or DEFAULT_REGIONS
        attitude_vals = axes.get("attitude_cluster") or DEFAULT_ATTITUDES

        combos = list(itertools.product(age_vals, income_vals, region_vals, attitude_vals))
        assignments: Dict[str, dict] = {}
        for idx, (age, income, region, attitude) in enumerate(combos):
            if idx >= cfg.persona_goal:
                break
            persona_id = f"dir_{idx + 1:04d}"
            attributes = {
                "age_band": age,
                "income_band": income,
                "region": region,
                "attitude_cluster": attitude,
            }
            focus = (
                f"{region} の {age} / {income} 層、{cfg.domain} で {attitude.replace('_', ' ')} を重視"
            )
            assignments[persona_id] = {"attributes": attributes, "focus": focus}
        if not assignments:
            assignments["dir_0001"] = {
                "attributes": {
                    "age_band": DEFAULT_AGE_BANDS[0],
                    "income_band": DEFAULT_INCOME_BANDS[0],
                    "region": DEFAULT_REGIONS[0],
                    "attitude_cluster": DEFAULT_ATTITUDES[0],
                },
                "focus": f"{cfg.domain} に関する汎用ペルソナ",
            }
        return direction_yaml, assignments

    def _build_direction_yaml(self, cfg: MassPersonaJobConfig, axes: Dict[str, List[str]]) -> str:
        lines = [
            "version: 1",
            f"project: \"{cfg.project_name}\"",
            f"domain: \"{cfg.domain}\"",
            f"language: \"{cfg.language}\"",
            f"persona_goal: {cfg.persona_goal}",
            "axes:",
        ]
        for axis_name, values in axes.items():
            lines.append(f"  - name: {axis_name}")
            values_literal = ", ".join(values)
            lines.append(f"    values: [{values_literal}]")
        if cfg.direction_must_cover:
            lines.append("coverage:")
            lines.append(f"  must_cover: {cfg.direction_must_cover}")
        if cfg.notes:
            lines.append("notes: |")
            for line in cfg.notes.splitlines():
                lines.append(f"  {line}")
        return "\n".join(lines) + "\n"

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = (self.root_dir / path).resolve()
        return path

    def _column_letter_to_index(self, column: Optional[str]) -> int:
        col = (column or "A").strip().upper()
        if not col or not col.isalpha():
            raise ValueError(f"Invalid column reference: {column}")
        index = 0
        for char in col:
            index = index * 26 + (ord(char) - ord("A") + 1)
        return index - 1
