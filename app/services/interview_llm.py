from __future__ import annotations

import asyncio
import inspect
import json
import os
import random
import textwrap
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from ..models import InterviewJobConfig


AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
INCOME_BANDS = ["low", "mid", "high"]
REGIONS = ["JP_Kanto", "JP_Kansai", "US_West", "US_East", "EU_West"]
ATTITUDE_CLUSTERS = [
    "value_seeker",
    "premium_oriented",
    "eco_conscious",
    "trend_follower",
    "sensitive_skin",
]

GEMINI_DIRECTION_MODEL = "gemini-pro-latest"
OPENAI_PERSONA_MODEL = "gpt-4.1"
GEMINI_INTERVIEW_MODEL = "gemini-flash-lite-latest"

PersonaProgressCallback = Callable[[int, Dict[str, Any]], Awaitable[None] | None]
TranscriptProgressCallback = Callable[[int, Dict[str, Any]], Awaitable[None] | None]


class LLMGenerationError(Exception):
    """Raised when an upstream LLM integration fails."""


@dataclass
class DirectionResult:
    yaml_text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonaBatchResult:
    personas: List[dict]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterviewBatchResult:
    transcripts: List[dict]
    meta: Dict[str, Any] = field(default_factory=dict)


async def generate_direction_brief(cfg: InterviewJobConfig, *, extra_notes: Optional[str] = None) -> DirectionResult:
    """Generate a project direction brief as YAML via Gemini, with deterministic fallback."""
    notes = extra_notes or cfg.notes or ""
    region_values = _region_choices(cfg)
    system_prompt = textwrap.dedent(
        """\
        You are a senior marketing insights strategist. Produce concise YAML configuration files
        that orchestrate synthetic consumer interviews. The YAML must be valid, stable, and free of commentary.
        Always include sections: version, project, domain, stimulus_mode, target_personas, language,
        coverage_policy (strategy + mandatory_axes + optional_axes), axes (list of axis name + allowed values),
        ethics (prohibited + style), and guidance (bullet list of interview focal points).
        Use double quotes only when required, prefer lowercase snake_case identifiers, and keep arrays inline when short.
        """
    )
    language_label = _language_descriptor(cfg)
    user_prompt = textwrap.dedent(
        f"""\
        Project name: {cfg.project_name}
        Domain / category: {cfg.domain}
        Target personas: {cfg.persona_count}
        Language for outputs: {language_label}
        Stimulus mode: {cfg.stimulus_mode}
        Persona template hint: {cfg.persona_template or "not provided"}
        Notes / tone guidance: {notes or "not provided"}
        Computed language code: {cfg.language}

        Enumerations:
          age_band: {', '.join(AGE_BANDS)}
          income_band: {', '.join(INCOME_BANDS)}
          region: {', '.join(region_values)}
          attitude_cluster: {', '.join(ATTITUDE_CLUSTERS)}

        Output:
          - Valid YAML only, no code fences.
          - Use `age_band`, `income_band`, `region`, `attitude_cluster` as axis names.
          - Provide 3-5 bullet points under `guidance` describing interview probing angles in {language_label}.
        """
    )

    try:
        text, usage = await _call_gemini(
            GEMINI_DIRECTION_MODEL,
            system_prompt,
            user_prompt,
            temperature=0.2,
            top_p=0.8,
            max_output_tokens=896,
            response_mime_type="text/plain",
            retries=2,
        )
        yaml_text = text.strip()
        if not yaml_text:
            raise LLMGenerationError("Gemini returned empty direction brief")
        return DirectionResult(
            yaml_text=yaml_text,
            meta={
                "source": "llm",
                "model": GEMINI_DIRECTION_MODEL,
                "usage": usage,
            },
        )
    except Exception as exc:  # noqa: BLE001
        fallback = _fallback_direction_yaml(cfg, notes=notes)
        return DirectionResult(
            yaml_text=fallback,
            meta={
                "source": "fallback",
                "model": None,
                "error": str(exc),
            },
        )


async def generate_persona_batch(
    cfg: InterviewJobConfig,
    direction_yaml: str,
    *,
    tribes: Optional[List[dict]] = None,
    progress_cb: PersonaProgressCallback | None = None,
) -> PersonaBatchResult:
    """Generate personas via OpenAI with per-item fallback."""
    tribe_profiles = tribes or []
    total = max(1, cfg.tribe_count * cfg.persona_per_tribe) if cfg.tribe_count and cfg.persona_per_tribe else cfg.persona_count
    cfg.persona_count = total
    if total <= 0:
        return PersonaBatchResult(personas=[], meta={"total": 0, "llm_count": 0, "fallback_count": 0})

    effective_concurrency = max(1, min(cfg.concurrency or 4, 6))
    semaphore = asyncio.Semaphore(effective_concurrency)
    progress_lock = asyncio.Lock()
    completed = 0

    personas: List[Optional[dict]] = [None] * total
    errors: List[str] = []
    llm_count = 0
    fallback_count = 0

    async def _record_progress(index: int, info: Dict[str, Any]) -> None:
        nonlocal completed
        async with progress_lock:
            completed += 1
            if progress_cb:
                await _invoke_callback(progress_cb, completed, info | {"index": index})

    async def _produce(index: int) -> None:
        nonlocal llm_count, fallback_count
        persona_id = f"persona_{index + 1:04d}"
        seed = cfg.persona_seed + index
        template_hint = cfg.persona_template or ""
        tribe_idx = index // max(1, cfg.persona_per_tribe)
        tribe_profile = tribe_profiles[tribe_idx] if tribe_idx < len(tribe_profiles) else None
        persona_sequence = (index % max(1, cfg.persona_per_tribe)) + 1
        async with semaphore:
            try:
                llm_payload, usage = await _generate_persona_with_llm(
                    cfg,
                    direction_yaml=direction_yaml,
                    persona_id=persona_id,
                    index=index,
                    seed=seed,
                    template_hint=template_hint,
                    tribe_profile=tribe_profile,
                    tribe_id=tribe_idx + 1 if tribe_profile else None,
                    persona_seq=persona_sequence,
                )
                persona = _merge_persona_payload(
                    cfg,
                    llm_payload,
                    index=index,
                    seed=seed,
                    persona_id=persona_id,
                    template_hint=template_hint,
                    tribe_profile=tribe_profile,
                    tribe_id=tribe_idx + 1 if tribe_profile else None,
                    persona_seq=persona_sequence,
                )
                persona["tribe_id"] = tribe_idx + 1 if tribe_profile else persona.get("tribe_id", tribe_idx + 1)
                persona["persona_sequence"] = persona_sequence
                meta = {
                    "source": "llm",
                    "model": OPENAI_PERSONA_MODEL,
                    "usage": usage,
                }
                if persona.get("_warnings"):
                    meta["warnings"] = persona.pop("_warnings")
                persona["meta"] = meta
                llm_count += 1
                personas[index] = persona
                await _record_progress(index, {"persona_id": persona_id, "source": "llm", "tribe_id": tribe_idx + 1 if tribe_profile else None})
            except Exception as exc:  # noqa: BLE001
                fallback_count += 1
                errors.append(f"{persona_id}: {exc}")
                persona = _fallback_persona_payload(
                    cfg,
                    index=index,
                    seed=seed,
                    template_hint=template_hint,
                    tribe_profile=tribe_profile,
                    tribe_id=tribe_idx + 1 if tribe_profile else None,
                    persona_seq=persona_sequence,
                    reason=str(exc),
                )
                persona["tribe_id"] = tribe_idx + 1 if tribe_profile else persona.get("tribe_id", tribe_idx + 1)
                persona["persona_sequence"] = persona_sequence
                personas[index] = persona
                await _record_progress(index, {"persona_id": persona_id, "source": "fallback", "tribe_id": tribe_idx + 1 if tribe_profile else None, "error": str(exc)})

    tasks = [asyncio.create_task(_produce(i)) for i in range(total)]
    await asyncio.gather(*tasks)

    finalized = [p for p in personas if p is not None]
    return PersonaBatchResult(
        personas=finalized,
        meta={
            "total": total,
            "llm_count": llm_count,
            "fallback_count": fallback_count,
            "errors": errors or None,
            "models": [OPENAI_PERSONA_MODEL] if llm_count else [],
        },
    )


async def generate_interview_batch(
    cfg: InterviewJobConfig,
    personas: Sequence[dict],
    stimuli: Sequence[str],
    *,
    questions: Optional[List[str]] = None,
    progress_cb: TranscriptProgressCallback | None = None,
) -> InterviewBatchResult:
    """Simulate interviews via Gemini with fallback transcripts."""
    if not personas or not stimuli:
        return InterviewBatchResult(transcripts=[], meta={"total": 0, "llm_count": 0, "fallback_count": 0})

    effective_concurrency = max(1, min(cfg.concurrency or 4, 6))
    semaphore = asyncio.Semaphore(effective_concurrency)
    progress_lock = asyncio.Lock()
    completed = 0

    transcripts: List[dict] = []
    errors: List[str] = []
    llm_count = 0
    fallback_count = 0

    async def _record_progress(info: Dict[str, Any]) -> None:
        nonlocal completed
        async with progress_lock:
            completed += 1
            if progress_cb:
                await _invoke_callback(progress_cb, completed, info)

    async def _produce(persona: dict, stimulus: str) -> None:
        nonlocal llm_count, fallback_count
        persona_id = persona.get("persona_id") or "unknown"
        seed = persona.get("seed", cfg.persona_seed)
        if questions:
            persona.setdefault("questions", questions)
        async with semaphore:
            try:
                llm_payload, usage = await _generate_interview_with_llm(cfg, persona, stimulus, questions=questions)
                transcript = _merge_transcript_payload(cfg, llm_payload, persona, stimulus)
                transcript["meta"] = {
                    "source": "llm",
                    "model": GEMINI_INTERVIEW_MODEL,
                    "usage": usage,
                    "seed": seed,
                }
                if questions:
                    transcript["questions"] = list(questions)
                transcripts.append(transcript)
                llm_count += 1
                await _record_progress({"persona_id": persona_id, "stimulus": stimulus, "source": "llm"})
            except Exception as exc:  # noqa: BLE001
                fallback_count += 1
                errors.append(f"{persona_id}/{stimulus}: {exc}")
                transcript = _fallback_transcript(cfg, persona, stimulus, reason=str(exc))
                if questions:
                    transcript["questions"] = list(questions)
                transcripts.append(transcript)
                await _record_progress({"persona_id": persona_id, "stimulus": stimulus, "source": "fallback", "error": str(exc)})

    tasks = [
        asyncio.create_task(_produce(persona, stimulus))
        for persona in personas
        for stimulus in stimuli
    ]
    await asyncio.gather(*tasks)

    # Preserve deterministic ordering by persona_id then stimulus
    transcripts.sort(key=lambda item: (item.get("persona_id", ""), item.get("stimulus", "")))

    return InterviewBatchResult(
        transcripts=transcripts,
        meta={
            "total": len(transcripts),
            "llm_count": llm_count,
            "fallback_count": fallback_count,
            "errors": errors or None,
            "models": [GEMINI_INTERVIEW_MODEL] if llm_count else [],
        },
    )


async def _generate_persona_with_llm(
    cfg: InterviewJobConfig,
    *,
    direction_yaml: str,
    persona_id: str,
    index: int,
    seed: int,
    template_hint: str,
    tribe_profile: Optional[dict] = None,
    tribe_id: Optional[int] = None,
    persona_seq: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    language_label = _language_descriptor(cfg)
    region_values = _region_choices(cfg)
    system_prompt = textwrap.dedent(
        """\
        You are a senior consumer insights architect. Produce buyer personas as structured JSON.
        All natural-language fields must be written in the requested language.
        Avoid stereotypes, keep tone respectful, and ensure motivations and frictions are concrete and testable.
        """
    )

    persona_schema = {
        "name": "persona_profile",
        "schema": {
            "type": "object",
            "properties": {
                "persona_id": {"type": "string"},
                "persona_name": {"type": "string"},
                "persona_type": {
                    "type": "object",
                    "properties": {
                        "age_band": {"type": "string"},
                        "income_band": {"type": "string"},
                        "region": {"type": "string"},
                        "attitude_cluster": {"type": "string"},
                    },
                    "required": ["age_band", "income_band", "region", "attitude_cluster"],
                    "additionalProperties": False,
                },
                "prompt_brief": {"type": "string"},
                "motivations": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 4,
                    "items": {"type": "string"},
                },
                "frictions": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 4,
                    "items": {"type": "string"},
                },
                "tone": {"type": "string"},
                "background": {"type": "string"},
                "quotes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["persona_type", "prompt_brief", "motivations", "frictions"],
            "additionalProperties": True,
        },
    }

    tribe_brief = json.dumps(tribe_profile or {}, ensure_ascii=False, indent=2)
    user_prompt = textwrap.dedent(
        f"""\
        Project: {cfg.project_name}
        Domain: {cfg.domain}
        Persona index: {index + 1} / {cfg.persona_count}
        Persona identifier: {persona_id}
        Random seed: {seed}
        Output language: {language_label}
        Language tag: {cfg.language}
        Template hint: {template_hint or "not provided"}
        Interview notes: {cfg.notes or "not provided"}
        Stimulus mode: {cfg.stimulus_mode}
        Tribe blueprint (JSON):
        {tribe_brief}
        Persona position within tribe: {persona_seq or ((index % max(1, cfg.persona_per_tribe)) + 1)} / {cfg.persona_per_tribe}

        Choose categorical attributes from these enumerations only:
          age_band: {', '.join(AGE_BANDS)}
          income_band: {', '.join(INCOME_BANDS)}
          region: {', '.join(region_values)}
          attitude_cluster: {', '.join(ATTITUDE_CLUSTERS)}

        Direction overview (YAML):
        ---
        {direction_yaml.strip()}
        ---

        Requirements:
          - Ensure motivations/frictions reflect the persona's lived context.
          - Tie motivations to the domain ({cfg.domain}) and justify with context (workstyle, family, routines, etc.).
          - Tone must reflect {language_label} conversational norms.
          - Provide 0-2 short persona quotes capturing authentic voice (optional).
          - Keep total response concise (< 400 words).
          - Return JSON only, no explanations.
        """
    )

    content, usage = await _call_openai_chat(
        OPENAI_PERSONA_MODEL,
        system_prompt,
        user_prompt,
        json_schema=persona_schema,
        temperature=0.55,
        max_tokens=1024,
        retries=2,
    )
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise LLMGenerationError(f"Persona JSON decode failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMGenerationError("Persona payload is not an object")
    return payload, usage


async def generate_tribe_categories(
    cfg: InterviewJobConfig,
    utterances: List[str],
    *,
    headers: List[str],
    max_categories: int = 8,
    session_plan: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[dict], List[str], Dict[str, Any]]:
    if not utterances:
        return [], [], {}

    language_label = _language_descriptor(cfg)
    sample_block = "\n".join(f"- {text}" for text in utterances[:120])
    desired_count = max(4, min(max_categories, 8))
    header_list = headers or DEFAULT_TRIBE_HEADERS
    header_json = json.dumps(header_list, ensure_ascii=False)
    session_plan_text = ""
    if session_plan:
        session_lines: List[str] = []
        for entry in session_plan:
            start = entry.get("start_index")
            end = entry.get("end_index")
            session_id = entry.get("session_id")
            if not session_id:
                continue
            if start and end and start != end:
                label = f"Tribes {start}-{end}"
            elif start:
                label = f"Tribe {start}"
            else:
                label = "Tribe range"
            session_lines.append(f"  - {label}: {session_id}")
        if session_lines:
            session_plan_text = (
                "SessionID assignment (10件単位で同一IDを使用):\n" + "\n".join(session_lines)
            )

    system_prompt = textwrap.dedent(
        """\
        あなたは市場インサイトアーキテクトです。与えられた消費者の発話サンプルから、需要の幅が広く競合する"トライブ"カテゴリを定義してください。カテゴリは重複や表現違いを避け、調査やターゲティングにそのまま使える粒度にします。
        出力は JSON のみで、カテゴリ一覧には name / description / representative_needs / sample_phrases / fields(Tribe_SetUp 用の項目) / persona_guidance / common_questions を含めてください。
        """
    )

    tribe_schema = {
        "name": "tribe_categories",
        "schema": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": max_categories,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "representative_needs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "sample_phrases": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "fields": {
                                "type": "object",
                                "additionalProperties": True,
                            },
                            "persona_guidance": {
                                "type": "object",
                                "additionalProperties": True,
                            },
                            "question_focus": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "common_questions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["name", "description"],
                        "additionalProperties": True,
                    },
                },
                "common_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["categories"],
            "additionalProperties": False,
        },
    }

    session_plan_instructions = session_plan_text or "SessionID assignment: reuse the same SessionID value for each block of 10 tribes (1-10, 11-20, ...)."

    user_prompt = textwrap.dedent(
        f"""\
        Domain: {cfg.domain}
        Language: {language_label} (code: {cfg.language})
        Desired number of tribe categories: {desired_count}
        Persona count target: {cfg.persona_count}
        Columns for Tribe_SetUp sheet (B列以降): {header_json}
        {session_plan_instructions}

        発話サンプル (最大120件):
        {sample_block}

        要件:
          - カテゴリは重なりを避け、{cfg.domain} における態度・ニーズ・ライフスタイルの幅をカバーすること。
          - 各カテゴリの representative_needs は 2-4 件程度で箇条書きにすること。
          - sample_phrases には入力に含まれるフレーズを代表として抜粋すること。
          - fields オブジェクトには Tribe_SetUp シートの各列に対応する値を設定すること (列順は提示した配列に従う)。
          - fields.SessionID は提示された SessionID assignment に厳密一致させ、同一ブロック(10件)内で同じ値を用いること。
          - persona_guidance にはペルソナ生成時の補助情報（価値観、トーン、生活背景など）を含めること。
          - common_questions には全ペルソナ共通で使用する質問案を {cfg.questions_per_persona} 件含めること。
          - 生成は {language_label} で行うこと。
        """
    )

    message, usage = await _call_openai_chat(
        OPENAI_PERSONA_MODEL,
        system_prompt,
        user_prompt,
        json_schema=tribe_schema,
        temperature=0.35,
        max_tokens=1400,
        retries=2,
    )

    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise LLMGenerationError(f"Tribe category JSON decode failed: {exc}") from exc

    categories = payload.get("categories") if isinstance(payload, dict) else None
    if not isinstance(categories, list):
        categories = []
    shared_questions = payload.get("common_questions") if isinstance(payload, dict) else None
    if not isinstance(shared_questions, list):
        shared_questions = []
    meta_usage = usage or {}
    if session_plan:
        meta_usage = dict(meta_usage)
        meta_usage.setdefault("session_plan", session_plan)
    return categories, shared_questions, meta_usage


async def _generate_interview_with_llm(
    cfg: InterviewJobConfig,
    persona: dict,
    stimulus: str,
    *,
    questions: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    language_label = _language_descriptor(cfg)
    system_prompt = textwrap.dedent(
        """\
        You are conducting qualitative depth interviews. Produce structured JSON transcripts with thoughtful probes.
        The interviewer should sound like a professional moderator. The persona replies must reflect their profile faithfully.
        Avoid stereotyping and keep dialogue grounded in specific experiences.
        """
    )

    interview_schema = {
        "type": "object",
        "properties": {
            "rounds": {
                "type": "array",
                "minItems": cfg.max_rounds,
                "maxItems": cfg.max_rounds,
                "items": {
                    "type": "object",
                    "properties": {
                        "round": {"type": "integer"},
                        "interviewer": {"type": "string"},
                        "persona": {"type": "string"},
                    },
                    "required": ["round", "interviewer", "persona"],
                    "additionalProperties": False,
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["rounds"],
        "additionalProperties": True,
    }

    persona_brief = json.dumps(
        {
            "persona_type": persona.get("persona_type"),
            "motivations": persona.get("motivations"),
            "frictions": persona.get("frictions"),
            "tone": persona.get("tone"),
            "prompt_brief": persona.get("prompt_brief"),
            "background": persona.get("background"),
        },
        ensure_ascii=False,
        indent=2,
    )

    question_list = questions or persona.get("questions") or []
    if question_list:
        question_block = "\n".join(f"  {idx + 1}. {q}" for idx, q in enumerate(question_list))
        question_text = f"Shared question plan (use in order, adapt follow-up probes as needed):\n{question_block}\n"
    else:
        question_text = ""

    user_prompt = textwrap.dedent(
        f"""\
        Persona ID: {persona.get("persona_id")}
        Output language: {language_label}
        Language tag: {cfg.language}
        Domain context: {cfg.domain}
        Stimulus mode: {cfg.stimulus_mode}
        Stimulus description: {stimulus}

        Persona brief (JSON):
        {persona_brief}

        {question_text}
        Instructions:
          - Conduct exactly {cfg.max_rounds} rounds.
          - Use the shared question plan for the primary line of questioning, adding natural follow-up probes based on prior answers.
          - Each round must have a moderator question then persona answer.
          - Persona answers must reference motivations/frictions authentically, include concrete life situations, and avoid repetition.
          - Keep tone natural and consistent with the persona's speaking style if provided.
          - Provide a concise summary capturing overall sentiment (optional).
          - Reply with JSON only, matching the provided schema.
        """
    )

    content, usage = await _call_gemini(
        GEMINI_INTERVIEW_MODEL,
        system_prompt,
        user_prompt,
        temperature=0.4,
        top_p=0.85,
        max_output_tokens=1536,
        response_mime_type="application/json",
        response_schema=interview_schema,
        retries=2,
    )

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise LLMGenerationError(f"Interview JSON decode failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMGenerationError("Interview payload is not an object")
    return payload, usage


def _merge_persona_payload(
    cfg: InterviewJobConfig,
    llm_payload: Dict[str, Any],
    *,
    index: int,
    seed: int,
    persona_id: str,
    template_hint: str,
    tribe_profile: Optional[dict] = None,
    tribe_id: Optional[int] = None,
    persona_seq: Optional[int] = None,
) -> dict:
    """Blend LLM payload with deterministic defaults to ensure required structure."""
    defaults = _default_persona_payload(
        cfg,
        index=index,
        seed=seed,
        template_hint=template_hint,
        tribe_profile=tribe_profile,
        tribe_id=tribe_id,
        persona_seq=persona_seq,
    )
    defaults["persona_id"] = persona_id
    defaults["seed"] = seed
    defaults["_warnings"] = []

    persona_type = llm_payload.get("persona_type")
    if isinstance(persona_type, dict):
        allowed_map = {
            "age_band": AGE_BANDS,
            "income_band": INCOME_BANDS,
            "region": _region_choices(cfg),
            "attitude_cluster": ATTITUDE_CLUSTERS,
        }
        for key, allowed in allowed_map.items():
            value = persona_type.get(key)
            cleaned = _sanitize_choice(value, allowed)
            if cleaned:
                defaults["persona_type"][key] = cleaned
            else:
                defaults["_warnings"].append(f"{key} invalid -> default retained")
    else:
        defaults["_warnings"].append("persona_type missing -> defaults used")

    if isinstance(llm_payload.get("prompt_brief"), str) and llm_payload["prompt_brief"].strip():
        defaults["prompt_brief"] = llm_payload["prompt_brief"].strip()

    for key in ("motivations", "frictions"):
        values = llm_payload.get(key)
        if isinstance(values, list):
            cleaned = [str(item).strip() for item in values if isinstance(item, str) and item.strip()]
            if cleaned:
                defaults[key] = cleaned
            else:
                defaults["_warnings"].append(f"{key} empty -> defaults retained")
        else:
            defaults["_warnings"].append(f"{key} missing -> defaults retained")

    for optional_key in ("tone", "background", "persona_name"):
        value = llm_payload.get(optional_key)
        if isinstance(value, str) and value.strip():
            defaults[optional_key] = value.strip()

    quotes = llm_payload.get("quotes")
    if isinstance(quotes, list):
        cleaned_quotes = [str(item).strip() for item in quotes if isinstance(item, str) and item.strip()]
        if cleaned_quotes:
            defaults["quotes"] = cleaned_quotes

    if tribe_id is not None:
        defaults["tribe_id"] = tribe_id
    if persona_seq is not None:
        defaults["persona_sequence"] = persona_seq
    if isinstance(tribe_profile, dict):
        defaults.setdefault("tribe_profile", tribe_profile)
        session_id = tribe_profile.get("session_id")
        if session_id:
            defaults["session_id"] = session_id

    return defaults


def _merge_transcript_payload(
    cfg: InterviewJobConfig,
    llm_payload: Dict[str, Any],
    persona: dict,
    stimulus: str,
) -> dict:
    """Normalize LLM transcript payload."""
    fallback_rounds = _default_interview_rounds(cfg, persona, stimulus)
    warnings: List[str] = []

    rounds = llm_payload.get("rounds")
    normalized_rounds: List[dict] = []
    if isinstance(rounds, list):
        for idx, item in enumerate(rounds, start=1):
            if not isinstance(item, dict):
                warnings.append(f"round {idx} invalid -> fallback used")
                normalized_rounds.extend(fallback_rounds[min(idx - 1, len(fallback_rounds) - 1)])
                continue
            interviewer_text = str(item.get("interviewer", "")).strip()
            persona_text = str(item.get("persona", "")).strip()
            round_number = item.get("round") if isinstance(item.get("round"), int) else idx
            if not interviewer_text or not persona_text:
                warnings.append(f"round {idx} incomplete -> fallback used")
                normalized_rounds.extend(fallback_rounds[min(idx - 1, len(fallback_rounds) - 1)])
                continue
            normalized_rounds.extend(
                [
                    {"role": "interviewer", "round": round_number, "text": interviewer_text},
                    {"role": "persona", "round": round_number, "text": persona_text},
                ]
            )
    if not normalized_rounds:
        normalized_rounds = [
            turn for pair in fallback_rounds for turn in pair
        ]
        warnings.append("no valid rounds -> full fallback transcript used")

    summary = llm_payload.get("summary")
    transcript = {
        "persona_id": persona.get("persona_id"),
        "stimulus": stimulus,
        "turns": normalized_rounds,
    }
    if isinstance(summary, str) and summary.strip():
        transcript["summary"] = summary.strip()
    if warnings:
        transcript.setdefault("meta", {})
        transcript["meta"]["warnings"] = warnings
    return transcript


async def _call_gemini(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    response_mime_type: str,
    response_schema: Optional[dict] = None,
    timeout: float = 60.0,
    retries: int = 1,
) -> Tuple[str, Dict[str, Any]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMGenerationError("GEMINI_API_KEY is not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    generation_config: Dict[str, Any] = {
        "temperature": temperature,
        "topP": top_p,
        "maxOutputTokens": max_output_tokens,
        "responseMimeType": response_mime_type,
    }
    if response_schema is not None:
        generation_config["responseSchema"] = response_schema

    payload = {
        "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": generation_config,
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, params={"key": api_key}, json=payload)
            response.raise_for_status()
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            return text, usage
        except (httpx.HTTPStatusError, httpx.TransportError, KeyError, IndexError, TypeError, ValueError) as exc:
            last_error = exc
            retryable = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {429, 500, 502, 503, 504}
            if attempt < retries and retryable:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise LLMGenerationError(f"Gemini request failed: {exc}") from exc
    assert last_error is not None
    raise LLMGenerationError(f"Gemini request failed: {last_error}") from last_error


async def _call_openai_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    json_schema: Optional[dict] = None,
    temperature: float,
    max_tokens: int,
    timeout: float = 60.0,
    retries: int = 1,
) -> Tuple[str, Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMGenerationError("OPENAI_API_KEY is not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": json_schema}

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return message, usage
        except (httpx.HTTPStatusError, httpx.TransportError, KeyError, IndexError, TypeError, ValueError) as exc:
            last_error = exc
            retryable = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {429, 500, 502, 503, 504}
            if attempt < retries and retryable:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise LLMGenerationError(f"OpenAI request failed: {exc}") from exc
    assert last_error is not None
    raise LLMGenerationError(f"OpenAI request failed: {last_error}") from last_error


def _fallback_direction_yaml(cfg: InterviewJobConfig, *, notes: str) -> str:
    region_values = _region_choices(cfg)
    coverage_policy = textwrap.dedent(
        """\
        coverage_policy:
          strategy: stratified_maxdiversity
          mandatory_axes: ["age_band", "income_band", "region"]
          optional_axes: ["attitude_cluster"]
        """
    )
    guidance = [
        f"{cfg.domain}購入時の評価基準を掘り下げる",
        "ブランド信頼・情報源の役割を確認する",
        "生活リズムと使用シーンの関係を明らかにする",
    ]
    if cfg.language == "en":
        guidance = [
            f"Explore how they evaluate {cfg.domain} offerings",
            "Probe brand trust, validation sources, and peer influence",
            "Map habitual routines and concrete usage situations",
        ]
    guidance_block = "\n  - " + "\n  - ".join(guidance)

    ethics_block = textwrap.dedent(
        """\
        ethics:
          prohibited:
            - stereotypes
            - protected-attribute causal claims
          style: respectful, neutral, evidence-seeking
        """
    )
    axes_block = textwrap.dedent(
        f"""\
        axes:
          - name: age_band
            values: {AGE_BANDS}
          - name: income_band
            values: {INCOME_BANDS}
          - name: region
            values: {region_values}
          - name: attitude_cluster
            values: {ATTITUDE_CLUSTERS}
        """
    )
    notes_block = ""
    if notes:
        normalized = notes.replace("\r\n", "\n").strip()
        indented = textwrap.indent(normalized, "    ")
        notes_block = f"brief_notes: |\n{indented}\n"

    return textwrap.dedent(
        f"""\
        version: 1
        project: "{cfg.project_name}"
        domain: "{cfg.domain}"
        stimulus_mode: "{cfg.stimulus_mode}"
        target_personas: {cfg.persona_count}
        language: "{cfg.language}"
        {coverage_policy}
        {axes_block}
        {ethics_block}
        guidance:{guidance_block}
        {notes_block}"""
    ).strip() + "\n"


def _default_persona_payload(
    cfg: InterviewJobConfig,
    *,
    index: int,
    seed: int,
    template_hint: str,
    tribe_profile: Optional[dict] = None,
    tribe_id: Optional[int] = None,
    persona_seq: Optional[int] = None,
) -> dict:
    rng = random.Random(seed or (cfg.persona_seed + index))
    tribe_fields = (tribe_profile or {}).get("fields", {}) if isinstance(tribe_profile, dict) else {}

    def _field(*keys: str) -> Optional[str]:
        for key in keys:
            value = tribe_fields.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    region_options = _region_choices(cfg)
    age = _field("Age", "年代") or AGE_BANDS[index % len(AGE_BANDS)]
    income = _field("IncomeLevel", "Income", "年収") or INCOME_BANDS[rng.randrange(len(INCOME_BANDS))]
    region = _field("Region", "地域") or region_options[rng.randrange(len(region_options))]
    attitude = ATTITUDE_CLUSTERS[rng.randrange(len(ATTITUDE_CLUSTERS))]
    persona_id = f"persona_{index + 1:04d}"

    motivations = [
        f"Seeks {cfg.domain} solutions matching {attitude.replace('_', ' ')} values.",
        "Wants transparent ingredient stories and measurable outcomes.",
    ]
    frictions = [
        "Skeptical of generic marketing claims.",
        "Needs social proof from similar consumers.",
    ]
    if cfg.language == "ja":
        motivations = [
            f"{attitude.replace('_', ' ')}の価値観に沿った{cfg.domain}体験を重視している",
            "成分の透明性と生活へのフィット感を求めている",
        ]
        frictions = [
            "一般的なキャッチコピーへの信頼が薄い",
            "自分と似た生活者の実感や証言を重視する",
        ]

    prompt_hint = template_hint or tribe_fields.get("TribeName") or f"{region.replace('_', ' ')} resident balancing {cfg.domain} choices"

    payload = {
        "persona_id": persona_id,
        "seed": seed,
        "persona_type": {
            "age_band": age,
            "income_band": income,
            "region": region,
            "attitude_cluster": attitude,
        },
        "prompt_brief": prompt_hint,
        "motivations": motivations,
        "frictions": frictions,
        "tone": "curious and candid" if cfg.language == "en" else "率直で前向き",
        "background": "Urban professional balancing wellness and productivity.",
    }
    if cfg.language == "ja":
        payload["background"] = "都市部で働く忙しい生活者。効率とウェルネスの両立を模索している。"
    if tribe_id is not None:
        payload["tribe_id"] = tribe_id
    if persona_seq is not None:
        payload["persona_sequence"] = persona_seq
    return payload


def _fallback_persona_payload(
    cfg: InterviewJobConfig,
    *,
    index: int,
    seed: int,
    template_hint: str,
    tribe_profile: Optional[dict] = None,
    tribe_id: Optional[int] = None,
    persona_seq: Optional[int] = None,
    reason: Optional[str] = None,
) -> dict:
    payload = _default_persona_payload(
        cfg,
        index=index,
        seed=seed,
        template_hint=template_hint,
        tribe_profile=tribe_profile,
        tribe_id=tribe_id,
        persona_seq=persona_seq,
    )
    payload["meta"] = {"source": "fallback"}
    if reason:
        payload["meta"]["reason"] = reason
    return payload


def _default_interview_rounds(
    cfg: InterviewJobConfig,
    persona: dict,
    stimulus: str,
) -> List[List[dict]]:
    persona_type = persona.get("persona_type", {})
    persona_desc = (
        f"{persona_type.get('age_band', 'unknown')} / "
        f"{persona_type.get('income_band', 'unknown')} / "
        f"{persona_type.get('region', 'unknown')}"
    )
    motivations = persona.get("motivations") or []
    frictions = persona.get("frictions") or []
    shared_questions = persona.get("questions") or []
    rounds: List[List[dict]] = []
    for r in range(cfg.max_rounds):
        if shared_questions:
            base_question = shared_questions[r % len(shared_questions)]
            interviewer_text = base_question
        else:
            interviewer_text = (
                f"Round {r + 1}: Considering the concept '{stimulus}', "
                f"how does it resonate with your {cfg.domain} priorities?"
            )
        motivation_snippet = motivations[r % len(motivations)] if motivations else f"{cfg.domain} expectations."
        friction_snippet = frictions[r % len(frictions)] if frictions else "some unresolved concerns."
        persona_text = (
            f"As {persona_desc}, I value offerings that address {motivation_snippet}. "
            f"This concept {'aligns' if (r % 2 == 0) else 'raises questions'} because {friction_snippet}"
        )
        rounds.append(
            [
                {"role": "interviewer", "round": r + 1, "text": interviewer_text},
                {"role": "persona", "round": r + 1, "text": persona_text},
            ]
        )
    return rounds


def _fallback_transcript(
    cfg: InterviewJobConfig,
    persona: dict,
    stimulus: str,
    *,
    reason: Optional[str] = None,
) -> dict:
    rounds = _default_interview_rounds(cfg, persona, stimulus)
    turns = [turn for pair in rounds for turn in pair]
    transcript = {
        "persona_id": persona.get("persona_id"),
        "stimulus": stimulus,
        "turns": turns,
        "meta": {"source": "fallback"},
    }
    if reason:
        transcript["meta"]["reason"] = reason
    return transcript


def _sanitize_choice(value: Any, allowed: Sequence[str]) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned in allowed:
            return cleaned
    return None


def _language_descriptor(cfg: InterviewJobConfig) -> str:
    label = getattr(cfg, "language_label", None)
    if label:
        return label

    code = (cfg.language or "").lower()
    mapping = {
        "ja": "Japanese",
        "en": "English",
        "ko": "Korean",
        "zh": "Chinese",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "pt": "Portuguese",
        "it": "Italian",
        "hi": "Hindi",
        "id": "Indonesian",
        "th": "Thai",
    }
    if code in mapping:
        return mapping[code]
    return cfg.language or "English"


async def _invoke_callback(
    callback: Callable[[int, Dict[str, Any]], Awaitable[None] | None],
    completed: int,
    info: Dict[str, Any],
) -> None:
    result = callback(completed, info)
    if inspect.isawaitable(result):
        await result
def _region_choices(cfg: InterviewJobConfig) -> List[str]:
    if cfg.country_region:
        value = cfg.country_region.strip()
        if value:
            return [value]
    return REGIONS
