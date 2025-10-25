from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from itertools import product
from math import prod
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import httpx

from .models import (
    TribeInterviewJobConfig,
    TribeInterviewJobProgress,
    TribeInterviewJobResponse,
    TribeInterviewJobStatus,
    TribeInterviewMode,
    TribeInterviewStage,
)
from .services.google_sheets import (
    GoogleSheetsError,
    batch_update_values,
    clear_sheet,
    column_index_to_a1,
    extract_spreadsheet_id,
    write_column_values,
    write_row_values,
    write_table_with_headers,
)
from .services.llm_embedding import call_gemini_embedding, call_openai_embedding
from .services.llm_persona import call_gemini_flash_json, call_openai_persona
from .services.llm_utils import call_gemini_json


TRIBE_ATTRIBUTE_FIELDS: Sequence[tuple[str, str]] = (
    ("gender", "Gender"),
    ("age", "Age"),
    ("region", "Region"),
    ("income_level", "年収レベル (Income Level)"),
    ("family_structure", "家族形態 (Family Structure)"),
    ("living_environment", "居住環境 (Living Environment)"),
    ("job_type", "職業タイプ (Job Type)"),
    ("time_value", "時間価値観 (Time Value)"),
    ("consumption_value", "消費価値観 (Consumption Value)"),
    ("social_value", "社会性価値観 (Social Value)"),
    ("decision_criteria", "判断基準 (Decision Criteria)"),
    ("lifes_focus", "人生の主たる関心事 (Life's Focus)"),
    ("motivation_source", "モチベーション源泉 (Motivation Source)"),
    ("novelty_attitude", "新奇性への態度 (Attitude to Novelty)"),
    ("self_perception", "自己認識タイプ (Self-Perception)"),
    ("emotional_control", "感情コントロール (Emotional Control)"),
    ("primary_info_source", "主たる情報源 (Primary Info Source)"),
    ("purchase_channel", "購買チャネル (Purchase Channel)"),
    ("digital_literacy", "デジタル習熟度 (Digital Literacy)"),
    ("health_behavior", "健康への投資行動 (Health Behavior)"),
    ("communication_tool", "コミュニケーション手段 (Communication Tool)"),
)


TRIBE_COMBINATION_KEYS: Sequence[str] = ("gender", "age", "region", "income_level")


TRIBE_MECE_OPTIONS: dict[str, Sequence[str]] = {
    "gender": ("女性", "男性", "ノンバイナリー", "ジェンダーフルイド", "その他"),
    "age": (
        "20代前半",
        "20代後半",
        "30代前半",
        "30代後半",
        "40代前半",
        "40代後半",
        "50代前半",
        "50代後半",
        "60代以上",
    ),
    "region": (
        "首都圏都市部",
        "首都圏郊外",
        "地方主要都市",
        "地方郊外",
        "地方農村",
    ),
    "income_level": (
        "低所得層",
        "中所得層",
        "中高所得層",
        "高所得層",
        "富裕層",
    ),
}


class TribeInterviewJobManager:
    """Manager orchestrating the tribe → persona → interview pipeline."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir / "tribe_interview"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._jobs: dict[str, TribeInterviewJobProgress] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    async def create_job(self, config: TribeInterviewJobConfig) -> TribeInterviewJobResponse:
        """Register a new tribe-interview job and persist its initial metadata."""

        job_id = uuid.uuid4().hex[:12]
        progress = TribeInterviewJobProgress(
            job_id=job_id,
            status=TribeInterviewJobStatus.pending,
            stage=TribeInterviewStage.pending,
            message="queued",
            artifacts=None,
        )

        async with self._lock:
            self._jobs[job_id] = progress

        job_dir = self._job_dir(job_id)
        self._write_config(job_dir, config)
        self._write_summary(job_dir, progress)

        task = asyncio.create_task(self._run_job(job_id, config))
        self._tasks[job_id] = task

        def _cleanup_task(fut: asyncio.Future) -> None:  # noqa: ANN001
            self._tasks.pop(job_id, None)
            if fut.cancelled():
                return
            exc = fut.exception()
            if exc:
                logging.getLogger(__name__).error("Tribe interview job %s failed: %s", job_id, exc)

        task.add_done_callback(_cleanup_task)

        return TribeInterviewJobResponse(job_id=job_id, status=progress.status)

    def get_progress(self, job_id: str) -> TribeInterviewJobProgress:
        """Return job progress for the given job id or raise KeyError if unknown."""

        progress = self._jobs.get(job_id)
        if progress:
            return progress

        summary_path = self._job_dir(job_id) / "summary.json"
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                status = TribeInterviewJobStatus(data.get("status", TribeInterviewJobStatus.pending.value))
                stage = TribeInterviewStage(data.get("stage", TribeInterviewStage.pending.value))
                message = data.get("message")
                artifacts = data.get("artifacts")
                metrics = data.get("metrics")
                progress = TribeInterviewJobProgress(
                    job_id=job_id,
                    status=status,
                    stage=stage,
                    message=message,
                    artifacts=artifacts,
                    metrics=metrics,
                )
                self._jobs[job_id] = progress
                return progress
            except Exception as exc:  # pragma: no cover - defensive
                raise KeyError(job_id) from exc

        raise KeyError(job_id)

    async def _run_job(self, job_id: str, config: TribeInterviewJobConfig) -> None:
        job_dir = self._job_dir(job_id)
        try:
            await self._update_progress(
                job_id,
                status=TribeInterviewJobStatus.running,
                stage=TribeInterviewStage.tribe,
                message="Generating tribes...",
            )
            tribes = await self._generate_tribes(job_dir, config)
            await self._update_progress(
                job_id,
                status=TribeInterviewJobStatus.running,
                stage=TribeInterviewStage.tribe,
                message=f"Tribe generation completed ({len(tribes)} rows)",
                artifacts={"tribes": str((job_dir / "tribes.json").relative_to(self.base_dir))},
                metrics={"tribes": len(tribes)},
            )
            await self._update_progress(
                job_id,
                status=TribeInterviewJobStatus.running,
                stage=TribeInterviewStage.combination,
                message="Generating tribe combinations...",
            )
            combinations = self._generate_combinations(tribes)
            self._write_combination_outputs(job_dir, config, combinations)
            await self._update_progress(
                job_id,
                status=TribeInterviewJobStatus.running,
                stage=TribeInterviewStage.combination,
                message=f"Combination generation completed ({len(combinations)} rows)",
                artifacts={
                    "combinations": str((job_dir / "tribe_combinations.json").relative_to(self.base_dir))
                },
                metrics={"combinations": len(combinations)},
            )
            await self._update_progress(
                job_id,
                status=TribeInterviewJobStatus.running,
                stage=TribeInterviewStage.persona,
                message="Generating persona prompts...",
            )
            persona_prompts = await self._generate_personas(job_dir, config, combinations)
            await self._update_progress(
                job_id,
                status=TribeInterviewJobStatus.running,
                stage=TribeInterviewStage.persona,
                message=f"Persona generation completed ({len(persona_prompts)} prompts)",
                artifacts={
                    "persona_prompts": str((job_dir / "persona_prompts.json").relative_to(self.base_dir))
                },
                metrics={"personas": len(persona_prompts)},
            )
            await self._update_progress(
                job_id,
                status=TribeInterviewJobStatus.running,
                stage=TribeInterviewStage.qa,
                message="Generating interview responses...",
            )
            qa_records = await self._generate_interviews(job_dir, config, combinations, persona_prompts)
            await self._update_progress(
                job_id,
                status=TribeInterviewJobStatus.running,
                stage=TribeInterviewStage.qa,
                message=f"Interview generation completed ({len(qa_records)} responses)",
                artifacts={
                    "qa_responses": str((job_dir / "qa_responses.json").relative_to(self.base_dir))
                },
                metrics={"qa_responses": len(qa_records)},
            )
            await self._update_progress(
                job_id,
                status=TribeInterviewJobStatus.running,
                stage=TribeInterviewStage.embedding,
                message="Generating embeddings...",
            )
            embedding_records = await self._generate_embeddings(job_dir, config, qa_records)
            await self._update_progress(
                job_id,
                status=TribeInterviewJobStatus.completed,
                stage=TribeInterviewStage.embedding,
                message="Pipeline completed",
                artifacts={
                    "qa_embeddings": str((job_dir / "qa_embeddings.json").relative_to(self.base_dir))
                },
                metrics={"qa_embeddings": len(embedding_records)},
            )
        except Exception as exc:  # noqa: BLE001 - propagate failure state
            await self._update_progress(
                job_id,
                status=TribeInterviewJobStatus.failed,
                stage=self._jobs[job_id].stage,
                message=str(exc),
            )

    async def _update_progress(
        self,
        job_id: str,
        *,
        status: TribeInterviewJobStatus | None = None,
        stage: TribeInterviewStage | None = None,
        message: str | None = None,
        artifacts: Dict[str, str] | None = None,
        metrics: Dict[str, int] | None = None,
    ) -> TribeInterviewJobProgress:
        async with self._lock:
            current = self._jobs.get(job_id)
            if not current:
                raise KeyError(job_id)
            merged_artifacts = dict(current.artifacts or {})
            if artifacts:
                merged_artifacts.update(artifacts)
            merged_metrics = dict(current.metrics or {})
            if metrics:
                merged_metrics.update(metrics)
            updated = TribeInterviewJobProgress(
                job_id=job_id,
                status=status or current.status,
                stage=stage or current.stage,
                message=message if message is not None else current.message,
                artifacts=merged_artifacts or None,
                metrics=merged_metrics or None,
            )
            self._jobs[job_id] = updated
        self._write_summary(self._job_dir(job_id), updated)
        return updated

    async def _generate_tribes(
        self,
        job_dir: Path,
        config: TribeInterviewJobConfig,
    ) -> List[Dict[str, str]]:
        prompt = self._build_tribe_prompt(config)

        last_error: Exception | None = None
        model_candidates = ("gemini-pro-latest", "gemini-flash-latest")
        backoff_base = 1.5
        for attempt in range(config.retry_limit):
            for model_name in model_candidates:
                try:
                    raw = await call_gemini_json(
                        prompt,
                        model=model_name,
                        timeout=90.0,
                    )
                    tribes = self._parse_tribe_response(raw, config)
                    self._write_tribe_outputs(job_dir, config, tribes)
                    return tribes
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    if model_name == "gemini-pro-latest" and exc.response.status_code in {429, 503}:
                        continue
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if model_name == "gemini-pro-latest":
                        continue
            await asyncio.sleep(backoff_base ** attempt + 0.5)
        raise RuntimeError(f"Tribe generation failed after {config.retry_limit} attempts: {last_error}")

    @staticmethod
    def _build_tribe_prompt(config: TribeInterviewJobConfig) -> str:
        attribute_list = "\n".join(
            f"- {header} (key: {key})" for key, header in TRIBE_ATTRIBUTE_FIELDS
        )
        mece_guidance = []
        for key, header in TRIBE_ATTRIBUTE_FIELDS:
            options = TRIBE_MECE_OPTIONS.get(key)
            if options:
                mece_guidance.append(f"- {header}: {', '.join(options)}")
        mece_block = "\n".join(mece_guidance)

        return (
            "You are an insights strategist creating distinct consumer tribes.\n"
            f"Product category: {config.product_category}\n"
            f"Country/Region context: {config.country_region}\n"
            f"Generate between 3 and {config.max_tribes} tribes. It is acceptable to return fewer than {config.max_tribes} segments if diversity would otherwise suffer.\n"
            "Ensure Gender, Age, Region, and Income Level selections are mutually exclusive and collectively cover the category needs.\n"
            "Use only the allowed value buckets below (do not invent new phrasings or combine ranges).\n"
            f"Allowed options:\n{mece_block}\n\n"
            "For each tribe, respond with a JSON object using the keys listed below. "
            "Values must be concise Japanese phrases (<= 50 characters) and must use exactly one of the allowed buckets when provided."
            "\nKeys:\n"
            f"{attribute_list}\n"
            "Return a JSON array of tribe objects with no extra commentary."
        )

    def _parse_tribe_response(
        self,
        raw: str,
        config: TribeInterviewJobConfig,
    ) -> List[Dict[str, str]]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:  # noqa: BLE001
            raise ValueError(f"Gemini returned non-JSON payload: {raw}") from exc

        if not isinstance(data, list) or not data:
            raise ValueError("Tribe response must be a non-empty JSON array")
        if len(data) > config.max_tribes:
            data = data[: config.max_tribes]

        normalized: List[Dict[str, str]] = []
        seen_keys: set[tuple[str, ...]] = set()
        key_subset = ("gender", "age", "region")

        for idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Tribe entry #{idx} is not an object: {item}")
            tribe: Dict[str, str] = {}
            for key, header in TRIBE_ATTRIBUTE_FIELDS:
                value = str(item.get(key, "")).strip()
                if not value:
                    raise ValueError(f"Tribe entry #{idx} missing value for '{key}'")
                tribe[key] = self._normalize_mece_value(key, value)
            duplicate_key = tuple(tribe[k] for k in key_subset)
            if duplicate_key in seen_keys:
                continue
            seen_keys.add(duplicate_key)
            normalized.append(tribe)

        if not normalized:
            raise ValueError("No valid tribe records generated")
        return normalized

    @staticmethod
    def _normalize_mece_value(key: str, raw_value: str) -> str:
        value = str(raw_value).strip()
        options = TRIBE_MECE_OPTIONS.get(key)
        if not options:
            return value
        if value in options:
            return value

        normalized = value.replace("〜", "~").replace("～", "~").replace("―", "-").replace("－", "-")
        parts = [part.strip() for part in re.split(r"[~\-]", normalized) if part.strip()]
        for part in parts:
            if part in options:
                return part

        replacements = {
            "女性層": "女性",
            "男性層": "男性",
            "ノンバイナリー層": "ノンバイナリー",
            "ジェンダーレス": "ジェンダーフルイド",
            "中産階級": "中所得層",
            "中流層": "中所得層",
            "中の上": "中高所得層",
            "高所得者": "高所得層",
            "富裕層以上": "富裕層",
        }
        mapped = replacements.get(value)
        if mapped and mapped in options:
            return mapped

        digit_match = re.match(r"(\d{2})代", value)
        if digit_match:
            prefix = digit_match.group(1)
            for option in options:
                if option.startswith(f"{prefix}代"):
                    return option

        for option in options:
            if option in value:
                return option

        allowed = ", ".join(options)
        raise ValueError(f"Value '{value}' for {key} must map to one of: {allowed}")

    def _write_tribe_outputs(
        self,
        job_dir: Path,
        config: TribeInterviewJobConfig,
        tribes: Sequence[Dict[str, str]],
    ) -> None:
        tribes_path = job_dir / "tribes.json"
        tribes_path.write_text(json.dumps(tribes, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            spreadsheet_id = extract_spreadsheet_id(config.spreadsheet_url)
        except GoogleSheetsError as exc:
            raise RuntimeError(f"スプレッドシートIDの取得に失敗しました: {exc}") from exc

        headers = [header for _, header in TRIBE_ATTRIBUTE_FIELDS]
        rows = [[tribe[key] for key, _ in TRIBE_ATTRIBUTE_FIELDS] for tribe in tribes]

        write_table_with_headers(
            spreadsheet_id,
            config.sheet_names.tribe_setup,
            headers,
            rows,
            start_row=1,
            start_col=2,
        )

        ids = [f"Tribe-{idx:02d}" for idx in range(1, len(rows) + 1)]
        write_column_values(
            spreadsheet_id,
            config.sheet_names.tribe_setup,
            column_index=1,
            start_row=1,
            values=["Tribe ID", *ids],
        )

    def _generate_combinations(self, tribes: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        """Expand core tribe attributes while keeping persona prompts tractable."""

        if not tribes:
            return []

        core_options: list[list[str]] = []
        for key in TRIBE_COMBINATION_KEYS:
            values = sorted({tribe.get(key, "").strip() for tribe in tribes if tribe.get(key)})
            if not values:
                values = [""]
            core_options.append(values)

        combo_count = prod(len(values) for values in core_options)
        if combo_count == 0:
            return []
        if combo_count > 2_000:
            logging.getLogger(__name__).warning(
                "Tribe combination space is still large (%s rows) after MECE reduction.",
                f"{combo_count:,}",
            )

        attribute_keys = [key for key, _ in TRIBE_ATTRIBUTE_FIELDS]
        combinations: List[Dict[str, str]] = []

        for combo_values in product(*core_options):
            core_mapping = dict(zip(TRIBE_COMBINATION_KEYS, combo_values))
            source = self._select_best_match(tribes, core_mapping)
            row: Dict[str, str] = {}
            for key in attribute_keys:
                if key in core_mapping:
                    row[key] = core_mapping[key]
                else:
                    row[key] = source.get(key, "")
            combinations.append(row)

        return combinations

    @staticmethod
    def _select_best_match(tribes: Sequence[Dict[str, str]], core_mapping: Dict[str, str]) -> Dict[str, str]:
        """Return the tribe whose attributes best align with the requested core mapping."""

        best: Optional[Dict[str, str]] = None
        best_score = -1
        for tribe in tribes:
            score = 0
            for key, target in core_mapping.items():
                if tribe.get(key) == target:
                    score += 1
            if score > best_score:
                best = tribe
                best_score = score
                if score == len(core_mapping):
                    break
        return best or tribes[0]

    def _write_combination_outputs(
        self,
        job_dir: Path,
        config: TribeInterviewJobConfig,
        combinations: Sequence[Dict[str, str]],
    ) -> None:
        combo_path = job_dir / "tribe_combinations.json"
        combo_path.write_text(json.dumps(combinations, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            spreadsheet_id = extract_spreadsheet_id(config.spreadsheet_url)
        except GoogleSheetsError as exc:
            raise RuntimeError(f"スプレッドシートIDの取得に失敗しました: {exc}") from exc

        sheet_name = config.sheet_names.tribe_combination
        headers = [header for _, header in TRIBE_ATTRIBUTE_FIELDS]
        rows = [[combo[key] for key, _ in TRIBE_ATTRIBUTE_FIELDS] for combo in combinations]

        clear_sheet(spreadsheet_id, sheet_name)

        full_headers = ["Combination ID", *headers]
        write_row_values(
            spreadsheet_id,
            sheet_name,
            row_index=1,
            start_col=1,
            values=full_headers,
        )

        if not rows:
            return

        batch_size = 500
        start_row = 2
        start_col_letter = column_index_to_a1(0)
        end_col_letter = column_index_to_a1(len(full_headers) - 1)
        quoted_sheet = f"'{sheet_name}'"

        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]
            payload: List[List[str]] = []
            for offset, values in enumerate(batch_rows, start=batch_start + 1):
                combo_id = f"Combo-{offset:02d}"
                payload.append([combo_id, *values])

            row_start = start_row + batch_start
            row_end = row_start + len(payload) - 1
            range_name = f"{quoted_sheet}!{start_col_letter}{row_start}:{end_col_letter}{row_end}"
            batch_update_values(
                spreadsheet_id,
                updates=[{"range": range_name, "values": payload}],
            )

    async def _generate_personas(
        self,
        job_dir: Path,
        config: TribeInterviewJobConfig,
        combinations: Sequence[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        persona_rows: List[List[str]] = []
        persona_records: List[Dict[str, str]] = []
        try:
            spreadsheet_id = extract_spreadsheet_id(config.spreadsheet_url)
        except GoogleSheetsError as exc:
            raise RuntimeError(f"スプレッドシートIDの取得に失敗しました: {exc}") from exc

        for index, combo in enumerate(combinations, start=1):
            for persona_idx in range(1, config.persona_per_combination + 1):
                prompt = self._build_persona_prompt(config, combo, persona_idx)
                data = await self._call_persona_llm(prompt, config)
                record = self._normalize_persona_response(data, combo, index, persona_idx)
                persona_records.append(record)
                persona_rows.append(self._persona_row(record))

        persona_path = job_dir / "persona_prompts.json"
        persona_path.write_text(json.dumps(persona_records, ensure_ascii=False, indent=2), encoding="utf-8")

        headers = [
            "Combination ID",
            "Persona ID",
            "Persona Name",
            "Summary",
            "Needs",
            "Tone",
            "Mode",
        ]
        write_table_with_headers(
            spreadsheet_id,
            config.sheet_names.persona_setup,
            headers,
            persona_rows,
            start_row=1,
            start_col=1,
        )
        return persona_records

    def _build_persona_prompt(
        self,
        config: TribeInterviewJobConfig,
        combination: Dict[str, str],
        persona_index: int,
    ) -> str:
        attributes = "\n".join(f"- {header}: {combination[key]}" for key, header in TRIBE_ATTRIBUTE_FIELDS)
        mode_desc = (
            "プロダクト評価に集中してください。" if config.mode == TribeInterviewMode.product else "タグラインへの反応に集中してください。"
        )
        base_prompt = (
            f"You are creating a synthetic persona blueprint.\n"
            f"Mode: {config.mode.value}\n"
            f"Instructions: {mode_desc}\n"
            "Persona attributes:"
            f"\n{attributes}\n"
        )
        if config.mode == TribeInterviewMode.product and config.product_detail:
            base_prompt += f"Relevant product detail:\n{config.product_detail}\n"
        if config.mode == TribeInterviewMode.communication and config.tagline_detail:
            base_prompt += f"Relevant tagline:\n{config.tagline_detail}\n"
        if config.persona_prompt_template:
            base_prompt += f"\nAdditional template guidance:\n{config.persona_prompt_template}\n"
        base_prompt += (
            "\nGenerate a JSON object with keys persona_name, summary, needs (array of 3 items), tone."
        )
        return base_prompt

    async def _call_persona_llm(
        self,
        prompt: str,
        config: TribeInterviewJobConfig,
    ) -> Dict[str, str]:
        last_error: Exception | None = None
        for attempt in range(config.retry_limit):
            try:
                return await call_gemini_flash_json(prompt, timeout=90.0)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                await asyncio.sleep(1.5 ** attempt)
        # Fallback to OpenAI
        for attempt in range(config.retry_limit):
            try:
                return await call_openai_persona(prompt, timeout=90.0)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                await asyncio.sleep(1.5 ** attempt)
        raise RuntimeError(f"Persona generation failed after retries: {last_error}")

    def _normalize_persona_response(
        self,
        data: Dict[str, object],
        combination: Dict[str, str],
        combo_index: int,
        persona_index: int,
    ) -> Dict[str, str]:
        try:
            persona_name = str(data.get("persona_name", "")).strip()
            summary = str(data.get("summary", "")).strip()
            needs_raw = data.get("needs", [])
            if not isinstance(needs_raw, list):
                raise ValueError("needs must be a list")
            needs_clean = [str(item).strip() for item in needs_raw if str(item).strip()]
            tone = str(data.get("tone", "")).strip()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid persona schema: {data}") from exc

        if len(needs_clean) < 3:
            needs_clean.extend([""] * (3 - len(needs_clean)))

        persona_id = f"Persona-{combo_index:02d}-{persona_index:02d}"
        combination_id = f"Combo-{combo_index:02d}"

        return {
            "combination_id": combination_id,
            "persona_id": persona_id,
            "persona_name": persona_name or persona_id,
            "summary": summary,
            "needs": "\n".join(needs_clean[:3]),
            "tone": tone,
            "mode": combination.get("mode", config.mode.value),
        }

    @staticmethod
    def _persona_row(record: Dict[str, str]) -> List[str]:
        return [
            record["combination_id"],
            record["persona_id"],
            record["persona_name"],
            record["summary"],
            record["needs"],
            record["tone"],
            record["mode"],
        ]

    async def _generate_interviews(
        self,
        job_dir: Path,
        config: TribeInterviewJobConfig,
        combinations: Sequence[Dict[str, str]],
        personas: Sequence[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        try:
            spreadsheet_id = extract_spreadsheet_id(config.spreadsheet_url)
        except GoogleSheetsError as exc:
            raise RuntimeError(f"スプレッドシートIDの取得に失敗しました: {exc}") from exc

        combination_map = {
            f"Combo-{idx:02d}": combo for idx, combo in enumerate(combinations, start=1)
        }

        questions = self._resolve_questions(config)
        total_columns = config.persona_per_combination * config.interviews_per_persona
        combination_answers: Dict[str, List[str]] = {
            combo_id: [] for combo_id in combination_map.keys()
        }
        qa_records: List[Dict[str, object]] = []
        personas_by_combo: Dict[str, List[Dict[str, str]]] = {}
        for persona in personas:
            personas_by_combo.setdefault(persona["combination_id"], []).append(persona)

        header_values = ["Combination ID"]
        for persona_slot in range(1, config.persona_per_combination + 1):
            for question_slot in range(1, config.interviews_per_persona + 1):
                header_values.append(f"P{persona_slot:02d}-Q{question_slot}")

        write_row_values(
            spreadsheet_id,
            config.sheet_names.qa_llm,
            row_index=1,
            start_col=1,
            values=header_values,
        )

        for persona in personas:
            combo_id = persona["combination_id"]
            combination = combination_map.get(combo_id)
            if combination is None:
                continue
            for idx in range(config.interviews_per_persona):
                question = questions[min(idx, len(questions) - 1)]
                prompt = self._build_interview_prompt(config, persona, combination, question)
                data = await self._call_interview_llm(prompt, config)
                answer, confidence = self._normalize_interview_response(data)
                formatted = (
                    f"{persona['persona_id']} Q{idx + 1}: {answer.strip()} "
                    f"(confidence {confidence:.2f})"
                )
                combination_answers.setdefault(combo_id, []).append(formatted)
                qa_records.append(
                    {
                        "combination_id": combo_id,
                        "persona_id": persona["persona_id"],
                        "question_index": idx + 1,
                        "question": question,
                        "answer": answer,
                        "confidence": confidence,
                    }
                )

        qa_path = job_dir / "qa_responses.json"
        qa_path.write_text(json.dumps(qa_records, ensure_ascii=False, indent=2), encoding="utf-8")

        for idx, combo_id in enumerate(combination_map.keys(), start=1):
            answers = combination_answers.get(combo_id, [])
            if len(answers) < total_columns:
                answers = answers + [""] * (total_columns - len(answers))
            else:
                answers = answers[:total_columns]
            row_values = [combo_id] + answers
            write_row_values(
                spreadsheet_id,
                config.sheet_names.qa_llm,
                row_index=idx + 1,
                start_col=1,
                values=row_values,
            )

        return qa_records

    async def _generate_embeddings(
        self,
        job_dir: Path,
        config: TribeInterviewJobConfig,
        qa_records: Sequence[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        try:
            spreadsheet_id = extract_spreadsheet_id(config.spreadsheet_url)
        except GoogleSheetsError as exc:
            raise RuntimeError(f"スプレッドシートIDの取得に失敗しました: {exc}") from exc

        embeddings: List[Dict[str, object]] = []
        row_map: Dict[str, List[str]] = {}
        total_columns = config.persona_per_combination * config.interviews_per_persona

        header_values = ["Combination ID"]
        for slot in range(total_columns):
            header_values.append(f"Embedding Slot {slot + 1:02d}")

        write_row_values(
            spreadsheet_id,
            config.sheet_names.qa_embedding,
            row_index=1,
            start_col=1,
            values=header_values,
        )

        for record in qa_records:
            text = str(record["answer"])
            vector = await self._call_embedding(text, config)
            embeddings.append({
                "combination_id": record["combination_id"],
                "persona_id": record["persona_id"],
                "question_index": record["question_index"],
                "embedding": vector,
            })
            row_map.setdefault(record["combination_id"], []).append(json.dumps(vector))

        embedding_path = job_dir / "qa_embeddings.json"
        embedding_path.write_text(json.dumps(embeddings, ensure_ascii=False, indent=2), encoding="utf-8")

        for idx, combo_id in enumerate(row_map.keys(), start=1):
            vectors = row_map[combo_id]
            if len(vectors) < total_columns:
                vectors = vectors + [""] * (total_columns - len(vectors))
            else:
                vectors = vectors[:total_columns]
            row_values = [combo_id] + vectors
            write_row_values(
                spreadsheet_id,
                config.sheet_names.qa_embedding,
                row_index=idx + 1,
                start_col=1,
                values=row_values,
            )

        return embeddings

    async def _call_embedding(
        self,
        text: str,
        config: TribeInterviewJobConfig,
    ) -> List[float]:
        last_error: Exception | None = None
        for attempt in range(config.retry_limit):
            try:
                return await call_gemini_embedding(text, timeout=60.0)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                await asyncio.sleep(1.5 ** attempt)
        for attempt in range(config.retry_limit):
            try:
                return await call_openai_embedding(text, timeout=60.0)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                await asyncio.sleep(1.5 ** attempt)
        raise RuntimeError(f"Embedding generation failed after retries: {last_error}")

    @staticmethod
    def _resolve_questions(config: TribeInterviewJobConfig) -> List[str]:
        if config.interview_questions:
            questions = [q.strip() for q in config.interview_questions if q.strip()]
            if questions:
                return questions
        if config.mode == TribeInterviewMode.product:
            return [
                "この製品を購入・継続利用する可能性と理由を教えてください。",
                "どのような改善や提案があれば、さらに魅力的に感じますか？",
                "購入をためらうとしたら、どんな点が不安になりますか？",
            ]
        return [
            "このタグラインにどんな第一印象を持ちましたか？",
            "どのような感情や情景が思い浮かびましたか？",
            "もっと響くようにするには、どんな表現に変えると良いと思いますか？",
        ]

    def _build_interview_prompt(
        self,
        config: TribeInterviewJobConfig,
        persona: Dict[str, str],
        combination: Dict[str, str],
        question: str,
    ) -> str:
        attributes = "\n".join(f"- {header}: {combination[key]}" for key, header in TRIBE_ATTRIBUTE_FIELDS)
        needs = persona.get("needs", "").replace("\n", ", ")
        detail_section = ""
        if config.mode == TribeInterviewMode.product and config.product_detail:
            detail_section = f"\nProduct detail:\n{config.product_detail}\n"
        elif config.mode == TribeInterviewMode.communication and config.tagline_detail:
            detail_section = f"\nTagline detail:\n{config.tagline_detail}\n"
        return (
            "You are role-playing a consumer persona.\n"
            f"Persona ID: {persona['persona_id']}\n"
            f"Persona Name: {persona['persona_name']}\n"
            f"Tone guideline: {persona['tone']}\n"
            f"Needs focus: {needs}\n"
            f"Tribe attributes:\n{attributes}\n"
            f"Mode: {config.mode.value}\n"
            f"Question: {question}\n"
            "Respond in Japanese, JSON only, with keys answer (string) and confidence (float 0-1)."
            f"{detail_section}"
        )

    async def _call_interview_llm(
        self,
        prompt: str,
        config: TribeInterviewJobConfig,
    ) -> Dict[str, object]:
        last_error: Exception | None = None
        for attempt in range(config.retry_limit):
            try:
                return await call_gemini_flash_json(prompt, timeout=90.0)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                await asyncio.sleep(1.5 ** attempt)
        for attempt in range(config.retry_limit):
            try:
                return await call_openai_persona(prompt, timeout=90.0)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                await asyncio.sleep(1.5 ** attempt)
        raise RuntimeError(f"Interview generation failed after retries: {last_error}")

    @staticmethod
    def _normalize_interview_response(data: Dict[str, object]) -> tuple[str, float]:
        try:
            answer = str(data.get("answer", "")).strip()
            confidence = float(data.get("confidence", 0.5))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid interview response: {data}") from exc
        confidence = max(0.0, min(1.0, confidence))
        return answer, confidence

    def _job_dir(self, job_id: str) -> Path:
        path = self.base_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_config(job_dir: Path, config: TribeInterviewJobConfig) -> None:
        config_path = job_dir / "config.json"
        config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _write_summary(job_dir: Path, progress: TribeInterviewJobProgress) -> None:
        summary_path = job_dir / "summary.json"
        summary_payload = {
            "job_id": progress.job_id,
            "status": progress.status.value,
            "stage": progress.stage.value,
            "message": progress.message,
            "artifacts": progress.artifacts,
            "metrics": progress.metrics,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    def job_dir(self, job_id: str) -> Path:
        return self._job_dir(job_id)

    def update_config(
        self,
        job_id: str,
        *,
        image_paths: Optional[List[str]] = None,
    ) -> None:
        job_dir = self._job_dir(job_id)
        config_path = job_dir / "config.json"
        if not config_path.exists():
            return
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if image_paths is not None:
            data["image_paths"] = image_paths
        config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    async def add_artifacts(self, job_id: str, artifacts: Dict[str, str]) -> TribeInterviewJobProgress:
        return await self._update_progress(job_id, artifacts=artifacts)


__all__ = ["TribeInterviewJobManager"]
