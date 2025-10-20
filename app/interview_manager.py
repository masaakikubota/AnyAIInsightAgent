from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    InterviewJobConfig,
    InterviewJobProgress,
    InterviewJobResponse,
    InterviewJobStatus,
    Category,
)
from .services.interview_llm import (
    AGE_BANDS,
    INCOME_BANDS,
    REGIONS,
    generate_direction_brief,
    generate_interview_batch,
    generate_persona_batch,
    generate_tribe_categories,
)
from .services.google_sheets import (
    GoogleSheetsError,
    batch_update_values,
    extract_spreadsheet_id,
    fetch_sheet_values,
    find_sheet,
)
from .services.embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts
from .services.has_scoring import score_utterance


ROOT_DIR = Path(__file__).resolve().parents[2]
TRIBE_CATEGORY_FILE = ROOT_DIR / "Tribe_Category"
DEFAULT_TRIBE_HEADERS = [
    "TribeName",
    "Gender",
    "Age",
    "Region",
    "IncomeLevel",
    "FamilyStructure",
    "Employment",
    "Lifestyle",
    "CoreValues",
    "Goals",
    "Frustrations",
    "BuyingTriggers",
    "PreferredChannels",
    "MediaTouchpoints",
    "PainPoints",
    "Motivations",
    "Opportunities",
    "Messaging",
    "RepresentativeQuote",
    "Notes",
    "SessionID",
]


logger = logging.getLogger(__name__)
HAS_LAMBDA = float(os.getenv("ANYAI_HAS_LAMBDA", "0.7"))


@dataclass
class InterviewJob:
    job_id: str
    config: InterviewJobConfig
    status: InterviewJobStatus = InterviewJobStatus.pending
    stage: str = "pending"
    total_personas: int = 0
    generated_personas: int = 0
    processed_transcripts: int = 0
    message: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0
    task: Optional[asyncio.Task] = None
    tribes: List[dict] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)


class InterviewJobManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir / "interview"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, InterviewJob] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, job_id: str, cfg: InterviewJobConfig) -> InterviewJobResponse:
        async with self._lock:
            job = InterviewJob(job_id=job_id, config=cfg)
            self.jobs[job_id] = job
        job_dir = self._job_dir(job_id)
        (job_dir / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
        job.task = asyncio.create_task(self._run_job(job))
        return InterviewJobResponse(job_id=job_id, status=job.status)

    def get_progress(self, job_id: str) -> InterviewJobProgress:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return InterviewJobProgress(
            job_id=job.job_id,
            status=job.status,
            stage=job.stage,
            total_personas=job.total_personas,
            generated_personas=job.generated_personas,
            processed_transcripts=job.processed_transcripts,
            message=job.message,
            artifacts=dict(job.artifacts) if job.artifacts else None,
        )

    def _job_dir(self, job_id: str) -> Path:
        path = self.base_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _session_id_for_index(self, job: InterviewJob, idx: int) -> str:
        """Derive a deterministic SessionID grouped per 10 tribes."""
        prefix = (job.job_id or "JOB")[:8].upper()
        group = idx // 10 + 1
        return f"{prefix}-CTX{group:02d}"

    def _build_session_plan(self, job: InterviewJob, total_tribes: int) -> List[dict]:
        if total_tribes <= 0:
            return []
        plan: List[dict] = []
        for start in range(0, total_tribes, 10):
            end = min(total_tribes, start + 10)
            plan.append(
                {
                    "start_index": start + 1,
                    "end_index": end,
                    "session_id": self._session_id_for_index(job, start),
                }
            )
        return plan

    async def _run_job(self, job: InterviewJob) -> None:
        job.status = InterviewJobStatus.running
        job.started_at = time.time()
        job_dir = self._job_dir(job.job_id)

        try:
            if job.config.manual_stimuli_images:
                stored = []
                for rel in job.config.manual_stimuli_images:
                    path = job_dir / rel
                    if path.exists():
                        stored.append(str(path.relative_to(self.base_dir)))
                if stored:
                    job.artifacts["manual_stimuli_images"] = ", ".join(stored)

            tribes, notes, questions = await self._generate_tribes(job, job_dir)
            job.tribes = tribes
            job.questions = questions

            await self._generate_direction_brief(job, job_dir, extra_notes=notes)
            await self._generate_personas(job, job_dir)
            await self._simulate_interviews(job, job_dir)
            job.status = InterviewJobStatus.completed
            job.stage = "completed"
            job.message = "Interview run completed successfully."
        except Exception as exc:  # noqa: BLE001
            job.status = InterviewJobStatus.failed
            job.stage = "failed"
            job.message = f"Interview pipeline failed: {exc}"
        finally:
            job.finished_at = time.time()
            summary = {
                "job_id": job.job_id,
                "status": job.status.value,
                "stage": job.stage,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "total_personas": job.total_personas,
                "generated_personas": job.generated_personas,
                "processed_transcripts": job.processed_transcripts,
                "artifacts": job.artifacts,
                "message": job.message,
            }
            (job_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            job.artifacts.setdefault("summary", str((job_dir / "summary.json").relative_to(self.base_dir)))

    async def _generate_direction_brief(self, job: InterviewJob, job_dir: Path, extra_notes: Optional[str] = None) -> None:
        job.stage = "direction_design"
        job.message = "方向性ブリーフを生成しています…"
        result = await generate_direction_brief(job.config, extra_notes=extra_notes)
        path = job_dir / "direction.yaml"
        path.write_text(result.yaml_text, encoding="utf-8")
        meta_path = job_dir / "direction_meta.json"
        meta_path.write_text(json.dumps(result.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        job.artifacts["direction"] = str(path.relative_to(self.base_dir))
        job.artifacts["direction_meta"] = str(meta_path.relative_to(self.base_dir))
        if result.meta.get("source") == "llm":
            job.message = "Direction brief generated via Gemini."
        else:
            error = result.meta.get("error") or "LLM unavailable"
            job.message = f"Direction brief fallback used ({error})."

    async def _generate_tribes(self, job: InterviewJob, job_dir: Path) -> Tuple[List[dict], Optional[str], List[str]]:
        cfg = job.config
        job.stage = "tribe_setup"
        job.message = "トライブカテゴリを生成しています…"

        headers = list(self._load_tribe_headers())
        if "SessionID" not in headers:
            headers.append("SessionID")

        utterances: List[str] = []
        if cfg.enable_tribe_learning and cfg.utterance_csv_path:
            csv_path = Path(cfg.utterance_csv_path)
            if not csv_path.is_absolute():
                csv_path = job_dir / csv_path
            if csv_path.exists():
                utterances = self._read_utterances_from_csv(csv_path, limit=400)
            else:
                job.message = f"トライブカテゴリ生成用CSVが見つかりません: {csv_path}"

        if not utterances:
            manual = self._parse_manual_stimuli(cfg.stimuli_source)
            if manual:
                utterances = manual[: min(len(manual), 120)]

        if not utterances:
            utterances = [
                f"私は{cfg.domain}を選ぶときに品質と安全性を重視している",
                f"{cfg.domain}ではコストパフォーマンスと使い勝手のバランスが大切",
                f"新しい{cfg.domain}を試すときは口コミとSNSを必ずチェックする",
            ]

        session_plan = self._build_session_plan(job, cfg.tribe_count)

        try:
            categories, questions, usage = await generate_tribe_categories(
                cfg,
                utterances,
                headers=headers,
                max_categories=cfg.tribe_count,
                session_plan=session_plan,
            )
        except Exception as exc:  # noqa: BLE001
            categories = self._fallback_tribes(job, headers)
            questions = self._default_questions(cfg, cfg.questions_per_persona)
            usage = {"error": str(exc)}
            job.message = (job.message or "") + (" | " if job.message else "") + "トライブカテゴリ生成でフォールバックを使用しました"

        tribes = self._normalize_tribes(job, categories, headers)

        if usage:
            payload = {
                "categories": categories,
                "usage": usage,
                "utterance_sample_size": len(utterances),
                "common_questions": questions,
            }
            out_path = job_dir / "tribe_categories.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            job.artifacts["tribe_categories"] = str(out_path.relative_to(self.base_dir))

        await self._write_tribes_to_sheet(job, tribes, headers)

        questions = list(questions or [])
        if len(questions) < cfg.questions_per_persona:
            questions.extend(self._default_questions(cfg, cfg.questions_per_persona - len(questions)))
        questions = questions[: cfg.questions_per_persona]

        summary_lines = []
        for tribe in tribes[: cfg.tribe_count]:
            name = tribe.get("name") or tribe.get("fields", {}).get(headers[0], f"Tribe {tribe['tribe_id']}")
            summary_lines.append(f"- {tribe['tribe_id']}: {name}")

        notes_parts: List[str] = []
        if summary_lines:
            notes_parts.append("Tribe overview:\n" + "\n".join(summary_lines))
        if questions:
            notes_parts.append("Shared interview questions:\n" + "\n".join(f"- {q}" for q in questions))
        notes = "\n\n".join(notes_parts) if notes_parts else None

        job.message = (job.message or "") + (" | " if job.message else "") + "トライブカテゴリを生成しました"
        job.artifacts["tribe_count"] = str(len(tribes))
        job.artifacts["persona_per_tribe"] = str(cfg.persona_per_tribe)
        return tribes, notes, questions

    def _load_tribe_headers(self) -> List[str]:
        if TRIBE_CATEGORY_FILE.exists():
            try:
                lines = [line.strip() for line in TRIBE_CATEGORY_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
                if lines:
                    return lines
            except Exception:
                pass
        return list(DEFAULT_TRIBE_HEADERS)

    def _read_utterances_from_csv(self, path: Path, limit: int = 200) -> List[str]:
        utterances: List[str] = []
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if not row:
                        continue
                    for cell in row:
                        text = str(cell).strip()
                        if text:
                            utterances.append(text)
                            break
                    if len(utterances) >= limit:
                        break
        except UnicodeDecodeError:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if not row:
                        continue
                    for cell in row:
                        text = str(cell).strip()
                        if text:
                            utterances.append(text)
                            break
                    if len(utterances) >= limit:
                        break
        return utterances

    def _normalize_tribes(self, job: InterviewJob, categories: List[dict], headers: List[str]) -> List[dict]:
        cfg = job.config
        total = max(1, cfg.tribe_count)
        headers_local = list(headers)
        if "SessionID" not in headers_local:
            headers_local.append("SessionID")

        normalized: List[dict] = []
        for idx in range(total):
            data = categories[idx] if idx < len(categories) else {}
            fields: Dict[str, Any] = {}
            raw_fields = {}
            if isinstance(data, dict):
                candidate = data.get("fields")
                if isinstance(candidate, dict):
                    raw_fields = candidate

            for header in headers_local:
                value = raw_fields.get(header)
                if value is None and isinstance(data, dict):
                    value = data.get(header) or data.get(header.lower())
                fields[header] = str(value).strip() if value is not None else ""

            name = ""
            if isinstance(data, dict):
                name = str(data.get("name") or data.get("label") or fields.get(headers_local[0]) or "").strip()
            if not name:
                name = f"Tribe {idx + 1}"
            fields[headers_local[0]] = name

            session_id = self._session_id_for_index(job, idx)
            fields["SessionID"] = session_id

            persona_guidance = {}
            if isinstance(data, dict):
                guidance = data.get("persona_guidance")
                if isinstance(guidance, dict):
                    persona_guidance = guidance

            normalized.append(
                {
                    "tribe_id": idx + 1,
                    "name": name,
                    "fields": fields,
                    "session_id": session_id,
                    "persona_guidance": persona_guidance,
                }
            )

        return normalized

    def _fallback_tribes(self, job: InterviewJob, headers: List[str]) -> List[dict]:
        cfg = job.config
        tribes: List[dict] = []
        genders = ["女性", "男性", "ノンバイナリー"] if cfg.language == "ja" else ["Female", "Male", "Non-binary"]
        lifestyles = ["都市部", "郊外", "地方"] if cfg.language == "ja" else ["Urban", "Suburban", "Rural"]
        opportunities = ["デジタル体験の最適化", "サステナブル訴求", "価格訴求"] if cfg.language == "ja" else ["Digital experience", "Sustainable positioning", "Value offering"]

        for idx in range(max(1, cfg.tribe_count)):
            fields = {header: "" for header in headers}
            if headers:
                fields[headers[0]] = f"Tribe {idx + 1}"
            if "Gender" in headers:
                fields["Gender"] = genders[idx % len(genders)]
            if "Age" in headers:
                fields["Age"] = AGE_BANDS[idx % len(AGE_BANDS)]
            if "Region" in headers:
                fields["Region"] = REGIONS[idx % len(REGIONS)]
            if "IncomeLevel" in headers:
                fields["IncomeLevel"] = INCOME_BANDS[idx % len(INCOME_BANDS)]
            if "Lifestyle" in headers:
                fields["Lifestyle"] = lifestyles[idx % len(lifestyles)]
            if "Opportunities" in headers:
                fields["Opportunities"] = opportunities[idx % len(opportunities)]
            session_id = self._session_id_for_index(job, idx)
            fields["SessionID"] = session_id

            tribes.append(
                {
                    "tribe_id": idx + 1,
                    "name": fields.get(headers[0], f"Tribe {idx + 1}") if headers else f"Tribe {idx + 1}",
                    "fields": fields,
                    "session_id": session_id,
                    "persona_guidance": {},
                }
            )
        return tribes

    def _default_questions(self, cfg: InterviewJobConfig, count: int) -> List[str]:
        templates = [
            "この{domain}のどんな点にもっと改善が必要だと感じますか?",
            "最近{domain}で心地よかった体験と、不満だった体験は何ですか?",
            "このコンセプトがあなたの生活にどう組み込まれるかイメージできますか?",
            "購入を決めるときに必ずチェックする情報源は何ですか?",
            "ブランドに期待するサポートやアフター体験は何ですか?",
        ] if cfg.language == "ja" else [
            "What aspects of {domain} do you feel still need improvement?",
            "Tell me about a great and a frustrating recent experience with {domain} products.",
            "Can you imagine how this concept would fit into your daily routine?",
            "What information sources do you trust before purchasing in {domain}?",
            "What support or aftercare do you expect from a brand in this space?",
        ]
        questions: List[str] = []
        for i in range(count):
            template = templates[i % len(templates)]
            questions.append(template.format(domain=cfg.domain))
        return questions

    def _build_tribe_sheet_rows(self, tribes: List[dict], headers: List[str]) -> Tuple[List[str], List[List[str]]]:
        header_order = list(headers)
        if "SessionID" not in header_order:
            header_order.append("SessionID")
        rows: List[List[str]] = []
        for tribe in tribes:
            fields = tribe.get("fields", {})
            rows.append([str(fields.get(header, "")) for header in header_order])
        return header_order, rows

    async def _write_tribes_to_sheet(self, job: InterviewJob, tribes: List[dict], headers: List[str]) -> None:
        cfg = job.config
        sheet_url = cfg.persona_sheet_url or cfg.stimuli_sheet_url
        if not sheet_url or not tribes:
            return

        try:
            spreadsheet_id = extract_spreadsheet_id(sheet_url)
        except GoogleSheetsError as exc:
            job.message = f"トライブ出力先シートの解析に失敗しました: {exc}"
            return

        try:
            tribe_sheet_name = (await asyncio.to_thread(find_sheet, spreadsheet_id, "Tribe_SetUp")).sheet_name
        except GoogleSheetsError:
            job.message = "Tribe_SetUp シートが見つからず、トライブを書き込めませんでした"
            return

        headers_local, rows = self._build_tribe_sheet_rows(tribes, headers)

        start_row = cfg.persona_start_row or 2
        if start_row < 1:
            start_row = 1

        start_col_index = 2
        start_col_letter = self._column_index_to_letter(start_col_index)
        end_col_letter = self._column_index_to_letter(start_col_index + len(headers_local) - 1)

        for batch_start in range(0, len(rows), 10):
            batch_end = min(batch_start + 10, len(rows))
            row_start = start_row + batch_start
            row_end = start_row + batch_end - 1
            values = rows[batch_start:batch_end]
            data = [
                {
                    "range": f"'{tribe_sheet_name}'!{start_col_letter}{row_start}:{end_col_letter}{row_end}",
                    "values": values,
                }
            ]
            try:
                await asyncio.to_thread(batch_update_values, spreadsheet_id, data)
            except GoogleSheetsError as exc:
                job.message = f"Tribe_SetUpへの書き込みに失敗しました: {exc}"
                return

        last_row = start_row + len(rows) - 1
        job.artifacts["tribe_sheet"] = f"{tribe_sheet_name}!{start_col_letter}{start_row}:{end_col_letter}{last_row}"
        job.message = (job.message or "") + (" | " if job.message else "") + "Tribe_SetUpにトライブ概要を出力しました"

    def _build_persona_summary(
        self,
        persona: dict,
        tribe: Optional[dict],
        questions: Optional[List[str]],
        cfg: InterviewJobConfig,
    ) -> str:
        persona_type = persona.get("persona_type", {})
        summary_parts: List[str] = []
        summary_parts.append(
            "Profile: "
            + " / ".join(
                filter(
                    None,
                    [
                        persona_type.get("age_band"),
                        persona_type.get("income_band"),
                        persona_type.get("region"),
                    ],
                )
            )
        )
        if tribe:
            tribe_name = tribe.get("name") or tribe.get("fields", {}).get("TribeName")
            if tribe_name:
                summary_parts.append(f"Tribe: {tribe_name}")
        motivations = persona.get("motivations") or []
        frictions = persona.get("frictions") or []
        if motivations:
            summary_parts.append("Motivations: " + "; ".join(motivations))
        if frictions:
            summary_parts.append("Frictions: " + "; ".join(frictions))
        tone = persona.get("tone")
        if tone:
            summary_parts.append(f"Tone: {tone}")
        background = persona.get("background")
        if background:
            summary_parts.append(f"Background: {background}")
        quote = persona.get("quotes")
        if quote:
            summary_parts.append("Quote: " + str(quote[0]))
        if questions:
            summary_parts.append("Shared Questions:")
            summary_parts.extend(f"- {q}" for q in questions)
        return "\n".join(summary_parts)

    async def _generate_personas(self, job: InterviewJob, job_dir: Path) -> None:
        job.stage = "persona_factory"
        job.total_personas = max(1, job.config.tribe_count * job.config.persona_per_tribe)
        job.config.persona_count = job.total_personas
        direction_path = job_dir / "direction.yaml"
        direction_text = direction_path.read_text(encoding="utf-8") if direction_path.exists() else ""
        total = job.total_personas

        if not job.tribes:
            job.tribes = self._normalize_tribes(job, [], self._load_tribe_headers())

        async def _on_progress(completed: int, info: Dict[str, Any]) -> None:
            job.generated_personas = completed
            descriptor = ""
            if info.get("persona_id") and info.get("source"):
                descriptor = f"{info['persona_id']} via {info['source']}"
            if total:
                job.message = f"ペルソナ生成中 {completed}/{total}" + (f" ({descriptor})" if descriptor else "")
            else:
                job.message = f"ペルソナ生成中 {completed}" + (f" ({descriptor})" if descriptor else "")

        result = await generate_persona_batch(job.config, direction_text, tribes=job.tribes, progress_cb=_on_progress)
        personas = result.personas
        job.generated_personas = len(personas)
        path = job_dir / "persona_catalog.json"
        path.write_text(json.dumps(personas, indent=2, ensure_ascii=False), encoding="utf-8")
        job.artifacts["persona_catalog"] = str(path.relative_to(self.base_dir))

        meta_path = job_dir / "persona_meta.json"
        meta_path.write_text(json.dumps(result.meta, indent=2, ensure_ascii=False), encoding="utf-8")
        job.artifacts["persona_meta"] = str(meta_path.relative_to(self.base_dir))

        llm_count = result.meta.get("llm_count", 0)
        fallback_count = result.meta.get("fallback_count", 0)
        errors = result.meta.get("errors")
        if errors:
            job.message = (
                f"ペルソナ生成完了 (LLM {llm_count}, fallback {fallback_count}). "
                f"latest warning: {errors[-1]}"
            )
        else:
            job.message = f"ペルソナ生成完了 (LLM {llm_count}, fallback {fallback_count})."

        if job.questions:
            for persona in personas:
                persona.setdefault("questions", job.questions)

        await self._write_personas_to_sheet(job, personas, job.tribes)

    async def _simulate_interviews(self, job: InterviewJob, job_dir: Path) -> None:
        job.stage = "interview_runner"
        personas_path = job_dir / "persona_catalog.json"
        if not personas_path.exists():
            raise FileNotFoundError("persona_catalog.json missing")
        personas: List[dict] = json.loads(personas_path.read_text(encoding="utf-8"))
        stimuli = await self._extract_stimuli(job)
        total_pairs = len(personas) * len(stimuli)

        async def _on_progress(completed: int, info: Dict[str, Any]) -> None:
            job.processed_transcripts = completed
            descriptor = ""
            if info.get("persona_id") and info.get("stimulus") and info.get("source"):
                descriptor = f"{info['persona_id']} × {info['stimulus']} via {info['source']}"
            if total_pairs:
                job.message = f"インタビュー生成中 {completed}/{total_pairs}" + (f" ({descriptor})" if descriptor else "")
            else:
                job.message = f"インタビュー生成中 {completed}" + (f" ({descriptor})" if descriptor else "")

        result = await generate_interview_batch(
            job.config,
            personas,
            stimuli,
            questions=job.questions,
            progress_cb=_on_progress,
        )
        transcripts = result.transcripts
        job.processed_transcripts = len(transcripts)
        await self._apply_hybrid_scores(job, transcripts)
        path = job_dir / "interview_transcripts.json"
        path.write_text(json.dumps(transcripts, indent=2, ensure_ascii=False), encoding="utf-8")
        job.artifacts["transcripts"] = str(path.relative_to(self.base_dir))
        meta_path = job_dir / "interview_meta.json"
        meta_path.write_text(json.dumps(result.meta, indent=2, ensure_ascii=False), encoding="utf-8")
        job.artifacts["interview_meta"] = str(meta_path.relative_to(self.base_dir))

        llm_count = result.meta.get("llm_count", 0)
        fallback_count = result.meta.get("fallback_count", 0)
        errors = result.meta.get("errors")
        if errors:
            job.message = (
                f"インタビュー生成完了 (LLM {llm_count}, fallback {fallback_count}). "
                f"latest warning: {errors[-1]}"
            )
        else:
            job.message = f"インタビュー生成完了 (LLM {llm_count}, fallback {fallback_count})."
        await self._generate_response_embeddings(job, transcripts)
        await self._write_qa_to_sheet(job, transcripts, personas, stimuli)

    async def _extract_stimuli(self, job: InterviewJob) -> List[str]:
        cfg = job.config
        job_dir = self._job_dir(job.job_id)
        manual_texts = self._parse_manual_stimuli(cfg.stimuli_source)

        if cfg.stimuli_sheet_url:
            try:
                stimuli = await self._read_stimuli_from_sheet(cfg)
                if stimuli:
                    job.message = f"Loaded {len(stimuli)} stimuli from Google Sheets."
                    return stimuli
                job.message = "Stimuli sheet had no values; falling back to manual entries."
            except Exception as exc:  # noqa: BLE001
                job.message = f"Stimuli sheet fallback: {exc}"

        stimuli: List[str] = []
        if manual_texts:
            stimuli.extend(manual_texts)

        image_entries: List[str] = []
        for rel in cfg.manual_stimuli_images or []:
            path = Path(rel)
            if not path.is_absolute():
                path = job_dir / rel
            if path.exists():
                image_entries.append(f"[Image Stimulus] {path.name} (path: {path.as_posix()})")

        if image_entries:
            stimuli.extend(image_entries)

        if stimuli:
            return stimuli
        return ["Baseline Concept"]

    async def _read_stimuli_from_sheet(self, cfg: InterviewJobConfig) -> List[str]:
        spreadsheet_id = extract_spreadsheet_id(cfg.stimuli_sheet_url or "")
        sheet_name = cfg.stimuli_sheet_name
        if not sheet_name:
            try:
                match = await asyncio.to_thread(find_sheet, spreadsheet_id, "Stimuli")
                sheet_name = match.sheet_name
            except GoogleSheetsError:
                sheet_name = "Sheet1"

        rows = await asyncio.to_thread(fetch_sheet_values, spreadsheet_id, sheet_name)
        if not rows:
            return []

        column_index = self._column_letter_to_index(cfg.stimuli_sheet_column)
        start_row = max(0, cfg.stimuli_sheet_start_row - 1)
        stimuli: List[str] = []
        for idx, row in enumerate(rows):
            if idx < start_row:
                continue
            cell = row[column_index] if column_index < len(row) else ""
            value = str(cell).strip()
            if value:
                stimuli.append(value)
        return stimuli

    def _parse_manual_stimuli(self, text: Optional[str]) -> List[str]:
        if not text:
            return []
        raw_items = [item.strip() for item in text.replace("\r", "\n").split("\n") if item.strip()]
        if not raw_items:
            return []
        stimuli: List[str] = []
        for item in raw_items:
            stimuli.extend([part.strip() for part in item.split(",") if part.strip()])
        return stimuli

    async def _generate_response_embeddings(self, job: InterviewJob, transcripts: List[dict]) -> None:
        cfg = job.config

        persona_entries: List[dict] = []
        response_texts: List[str] = []
        for transcript in transcripts:
            persona_turns = [
                turn.get("text", "")
                for turn in transcript.get("turns", [])
                if turn.get("role") == "persona"
            ]
            combined = "\n".join(filter(None, (text.strip() for text in persona_turns))).strip()
            if not combined:
                continue
            persona_entries.append(
                {
                    "persona_id": transcript.get("persona_id"),
                    "stimulus": transcript.get("stimulus"),
                    "response": combined,
                    "language": cfg.language,
                    "language_label": cfg.language_label,
                }
            )
            response_texts.append(combined)

        if not response_texts:
            summary = "Embeddings: persona応答が見つかりません"
            job.message = f"{job.message} | {summary}" if job.message else summary
            return

        try:
            embeddings = await embed_texts(response_texts, model=DEFAULT_EMBEDDING_MODEL)
        except Exception as exc:  # noqa: BLE001
            summary = f"Embeddings変換に失敗: {exc}"
            job.message = f"{job.message} | {summary}" if job.message else summary
            return

        for entry, vector in zip(persona_entries, embeddings):
            entry["embedding_model"] = DEFAULT_EMBEDDING_MODEL
            entry["embedding"] = vector

        out_path = self._job_dir(job.job_id) / "response_embeddings.json"
        out_path.write_text(json.dumps(persona_entries, ensure_ascii=False, indent=2), encoding="utf-8")
        job.artifacts["response_embeddings"] = str(out_path.relative_to(self.base_dir))
        summary = f"Embeddings: {len(persona_entries)}件を変換しました"
        job.message = f"{job.message} | {summary}" if job.message else summary

    def _build_persona_sheet_rows(self, job: InterviewJob, personas: List[dict], tribes: List[dict]) -> List[List[str]]:
        cfg = job.config
        tribe_map = {tribe.get("tribe_id"): tribe for tribe in tribes}

        persona_rows: List[List[str]] = []
        for tribe in sorted(tribes, key=lambda t: t.get("tribe_id", 0)):
            tribe_id = tribe.get("tribe_id")
            tribe_personas = [p for p in personas if (p.get("tribe_id") or 0) == tribe_id]
            tribe_personas.sort(key=lambda p: p.get("persona_sequence") or 0)
            for idx, persona in enumerate(tribe_personas, start=1):
                seq = persona.get("persona_sequence") or idx
                composed_id = f"{tribe_id}_{seq}"
                summary = self._build_persona_summary(persona, tribe_map.get(tribe_id), job.questions, cfg)
                persona_rows.append([composed_id, summary])

        unmatched = [p for p in personas if (p.get("tribe_id") or 0) not in tribe_map]
        for persona in unmatched:
            tribe_id = persona.get("tribe_id") or 0
            seq = persona.get("persona_sequence") or 1
            composed_id = f"{tribe_id}_{seq}" if tribe_id else persona.get("persona_id", "unknown")
            summary = self._build_persona_summary(persona, None, job.questions, cfg)
            persona_rows.append([composed_id, summary])

        return persona_rows

    async def _write_personas_to_sheet(self, job: InterviewJob, personas: List[dict], tribes: List[dict]) -> None:
        cfg = job.config
        sheet_url = cfg.persona_sheet_url or cfg.stimuli_sheet_url
        if not sheet_url or not personas:
            return

        try:
            spreadsheet_id = extract_spreadsheet_id(sheet_url)
        except GoogleSheetsError as exc:
            job.message = f"ペルソナ出力先シートの解析に失敗しました: {exc}"
            return

        try:
            persona_sheet_name = (await asyncio.to_thread(find_sheet, spreadsheet_id, "Persona_SetUp")).sheet_name
        except GoogleSheetsError:
            job.message = "Persona_SetUp シートが見つからないためペルソナ出力をスキップしました"
            return

        start_row = cfg.persona_start_row or 2
        if start_row < 1:
            start_row = 1

        persona_rows = self._build_persona_sheet_rows(job, personas, tribes)

        if not persona_rows:
            return

        start_col_letter = "A"
        end_col_letter = "B"

        for batch_start in range(0, len(persona_rows), 5):
            batch_end = min(batch_start + 5, len(persona_rows))
            row_start = start_row + batch_start
            row_end = start_row + batch_end - 1
            values = persona_rows[batch_start:batch_end]
            data = [
                {
                    "range": f"'{persona_sheet_name}'!{start_col_letter}{row_start}:{end_col_letter}{row_end}",
                    "values": values,
                }
            ]
            try:
                await asyncio.to_thread(batch_update_values, spreadsheet_id, data)
            except GoogleSheetsError as exc:
                job.message = f"Persona_SetUpへの書き込みに失敗しました: {exc}"
                return

        last_row = start_row + len(persona_rows) - 1
        job.artifacts["persona_sheet"] = f"{persona_sheet_name}!{start_col_letter}{start_row}:{end_col_letter}{last_row}"
        job.message = (job.message or "") + (" | " if job.message else "") + "Persona_SetUpにペルソナ概要を出力しました"

    async def _write_qa_to_sheet(
        self,
        job: InterviewJob,
        transcripts: List[dict],
        personas: List[dict],
        stimuli: List[str],
    ) -> None:
        cfg = job.config
        sheet_url = cfg.persona_sheet_url or cfg.stimuli_sheet_url
        if not sheet_url or not personas:
            return

        try:
            spreadsheet_id = extract_spreadsheet_id(sheet_url)
        except GoogleSheetsError as exc:
            job.message = f"QA出力先シートの解析に失敗しました: {exc}"
            return

        try:
            answer_sheet_name = (await asyncio.to_thread(find_sheet, spreadsheet_id, "QA_Answer")).sheet_name
        except GoogleSheetsError:
            job.message = "QA_Answer シートが見つからないためQA出力をスキップしました"
            return

        try:
            embedded_sheet_name = (await asyncio.to_thread(find_sheet, spreadsheet_id, "QA_Embedded")).sheet_name
        except GoogleSheetsError:
            job.message = "QA_Embedded シートが見つからないためQA出力をスキップしました"
            return

        start_row = cfg.persona_start_row or 2
        if start_row < 1:
            start_row = 1
        max_rounds = cfg.max_rounds or 1
        persona_order = [persona.get("persona_id") for persona in personas]
        persona_index_map = {pid: idx for idx, pid in enumerate(persona_order) if pid}
        if not persona_index_map:
            return

        answers_matrix: Dict[int, List[str]] = {idx: [""] * max_rounds for idx in range(len(persona_order))}
        embedded_matrix: Dict[int, List[str | float]] = {idx: [""] * max_rounds for idx in range(len(persona_order))}

        for transcript in transcripts:
            persona_id = transcript.get("persona_id")
            if persona_id not in persona_index_map:
                continue
            index = persona_index_map[persona_id]
            answers = answers_matrix.setdefault(index, [""] * max_rounds)
            persona_turns = [turn for turn in transcript.get("turns", []) if turn.get("role") == "persona"]
            slot = 0
            for turn in persona_turns:
                text = (turn.get("text") or "").strip()
                if not text:
                    continue
                while slot < max_rounds and answers[slot]:
                    slot += 1
                if slot >= max_rounds:
                    break
                answers[slot] = text
                slot += 1

        persona_count = len(persona_order)
        if persona_count == 0:
            return

        start_col_index = 2  # Column B
        end_col_index = start_col_index + max_rounds - 1
        start_col_letter = self._column_index_to_letter(start_col_index)
        end_col_letter = self._column_index_to_letter(end_col_index)

        for batch_start in range(0, persona_count, 5):
            batch_end = min(batch_start + 5, persona_count)
            if batch_start >= batch_end:
                continue
            answer_values = [answers_matrix[idx] for idx in range(batch_start, batch_end)]
            embedded_values = [embedded_matrix[idx] for idx in range(batch_start, batch_end)]
            row_start = start_row + batch_start
            row_end = start_row + batch_end - 1
            answer_range = f"'{answer_sheet_name}'!{start_col_letter}{row_start}:{end_col_letter}{row_end}"
            embedded_range = f"'{embedded_sheet_name}'!{start_col_letter}{row_start}:{end_col_letter}{row_end}"
            data = [
                {"range": answer_range, "values": answer_values},
                {"range": embedded_range, "values": embedded_values},
            ]
            try:
                await asyncio.to_thread(batch_update_values, spreadsheet_id, data)
            except GoogleSheetsError as exc:
                job.message = f"QAシートへの書き込みに失敗しました: {exc}"
                return

        job.artifacts["qa_answer_sheet"] = (
            f"{answer_sheet_name}!{start_col_letter}{start_row}:{end_col_letter}{start_row + persona_count - 1}"
        )
        job.artifacts["qa_embedded_sheet"] = (
            f"{embedded_sheet_name}!{start_col_letter}{start_row}:{end_col_letter}{start_row + persona_count - 1}"
        )
        job.message = f"QA結果をGoogle Sheets ({answer_sheet_name}, {embedded_sheet_name}) に書き込みました"

    async def _apply_hybrid_scores(self, job: InterviewJob, transcripts: List[dict]) -> None:
        if not transcripts:
            return
        for transcript in transcripts:
            concepts_data = transcript.get("concepts")
            analyses = transcript.get("analyses")
            if not concepts_data or not analyses:
                continue
            categories: List[Category] = []
            for entry in concepts_data:
                if isinstance(entry, dict):
                    name = str(entry.get("name") or entry.get("title") or entry.get("label") or "Concept").strip()
                    definition = str(
                        entry.get("definition")
                        or entry.get("detail")
                        or entry.get("text")
                        or name
                    ).strip()
                    detail = str(entry.get("detail") or entry.get("definition") or "").strip()
                else:
                    name = str(entry).strip()
                    definition = name
                    detail = ""
                categories.append(Category(name=name, definition=definition, detail=detail))
            if len(categories) != len(analyses):
                continue
            utterance_text = self._transcript_utterance_text(transcript)
            if not utterance_text:
                continue
            try:
                has_result = await score_utterance(utterance_text, categories, analyses)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "evt=interview_scoring_skipped persona_id=%s reason=%s",
                    transcript.get("persona_id"),
                    exc,
                )
                continue
            scoring_payload: Dict[str, Any] = {
                "absolute_scores": has_result.absolute_scores,
                "relative_rank_scores": has_result.relative_scores,
                "anchor_labels": [component.anchor for component in has_result.components],
                "concepts": [],
            }
            for idx, component in enumerate(has_result.components):
                concept_label = categories[idx].name or f"concept_{idx}"
                logger.debug(
                    "evt=anchor_parsed concept=%s anchor=%s context=interview",
                    concept_label,
                    component.anchor,
                )
                logger.debug(
                    "evt=score_components concept=%s anchor_w=%.2f similarity=%.6f r=%.6f p=%.6f lambda=%.2f final=%.6f context=interview",  # noqa: G004
                    concept_label,
                    component.anchor_weight,
                    component.similarity,
                    component.relative_score,
                    component.amplified_score,
                    HAS_LAMBDA,
                    component.final_score,
                )
                scoring_payload["concepts"].append(
                    {
                        "name": concept_label,
                        "anchor": component.anchor,
                        "anchor_weight": component.anchor_weight,
                        "similarity": component.similarity,
                        "relative_score": component.relative_score,
                        "amplified_score": component.amplified_score,
                        "absolute_score": has_result.absolute_scores[idx],
                        "relative_rank_score": has_result.relative_scores[idx],
                    }
                )
            transcript["scoring"] = scoring_payload

    def _transcript_utterance_text(self, transcript: Dict[str, Any]) -> str:
        summary = str(transcript.get("summary") or "").strip()
        if summary:
            return summary
        persona_turns = [
            str(turn.get("text") or "").strip()
            for turn in transcript.get("turns", [])
            if turn.get("role") == "persona"
        ]
        combined = "\n".join(text for text in persona_turns if text)
        return combined.strip()

    def _column_letter_to_index(self, column: str) -> int:
        col = (column or "A").strip().upper()
        if not col.isalpha():
            raise ValueError(f"Invalid column reference: {column}")
        index = 0
        for char in col:
            index = index * 26 + (ord(char) - ord("A") + 1)
        return index - 1

    def _column_index_to_letter(self, index: int) -> str:
        if index < 1:
            index = 1
        result = ""
        while index > 0:
            index, remainder = divmod(index - 1, 26)
            result = chr(65 + remainder) + result
        return result
