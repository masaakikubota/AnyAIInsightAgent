from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Literal

from pydantic import BaseModel, Field, conint, confloat, model_validator


class Provider(str, Enum):
    gemini = "gemini"
    openai = "openai"


class RunConfig(BaseModel):
    # CSV mapping (1-based indices for UX)
    utterance_col: conint(ge=1) = 3  # default C
    category_start_col: conint(ge=1) = 4  # default D
    name_row: conint(ge=1) = 2
    def_row: conint(ge=1) = 3
    detail_row: conint(ge=1) = 4
    start_row: conint(ge=1) = 5

    # Execution
    batch_size: conint(ge=1, le=50) = 10  # N categories per request (default 10)
    max_category_cols: conint(ge=1, le=10000) = 200  # Max columns to process per utterance
    max_retries: conint(ge=0, le=50) = 3
    concurrency: conint(ge=1, le=200) = 50
    auto_slowdown: bool = True
    timeout_sec: conint(ge=10, le=600) = 60
    sheet_chunk_rows: conint(ge=10, le=500) = 500
    chunk_row_limit: conint(ge=50, le=5000) = 500
    chunk_retry_limit: conint(ge=0, le=10) = 3
    cache_enabled: bool = True
    cache_ttl_seconds: conint(ge=60, le=604800) = 86400
    cache_max_entries: conint(ge=100, le=200000) = 10000
    pipeline_queue_size: Optional[conint(ge=1, le=2000)] = None
    validation_max_workers: conint(ge=1, le=64) = 4
    validation_worker_timeout_sec: conint(ge=1, le=300) = 30
    writer_flush_interval_sec: confloat(ge=0.1, le=30.0) = 6.0
    writer_flush_batch_size: conint(ge=1, le=500) = 50
    writer_retry_limit: conint(ge=1, le=10) = 5
    writer_retry_initial_delay_sec: confloat(ge=0.1, le=10.0) = 0.5
    writer_retry_backoff_multiplier: confloat(ge=1.0, le=5.0) = 2.0
    max_rows_per_job: Optional[conint(ge=100, le=1000000)] = 300
    slice_start_row: Optional[conint(ge=1)] = None
    slice_end_row: Optional[conint(ge=1)] = None
    parent_job_id: Optional[str] = None

    # Providers
    primary_provider: Provider = Provider.gemini
    fallback_provider: Provider = Provider.openai
    primary_model: Optional[str] = "gemini-flash-lite-latest"
    fallback_model: Optional[str] = "gpt-5-nano"
    enable_ssr: bool = True
    ssr_system_prompt: str = (
        "<prompt>\n"
        "    <role id=\"Semantic Inference Engine & Latent Intent Decoder\">\n"
        "        <spec>Noisy text (SNS, UGC, transcripts) analysis. Infer user's true goals, emotions, and unstated assumptions from fragmented data.</spec>\n"
        "        <ban>Surface-level keyword matching is strictly forbidden.</ban>\n"
        "    </role>\n"
        "\n"
        "    <mission>For a given utterance and N concepts, perform Semantic Similarity Rating (SSR) as described in AnyAI Scoring research (2025-10-14). First verbalize how similar each concept definition is to the latent intent of the utterance. Then provide succinct natural-language rationales that can be embedded for quantitative scoring.</mission>\n"
        "\n"
        "    <process>\n"
        "        <step id=\"1\" name=\"Normalize & Enrich\">\n"
        "            <task>Normalize slang, typos, jargon.</task>\n"
        "            <task>Interpret non-verbal cues (emojis, punctuation, irony markers) for emotional tone.</task>\n"
        "        </step>\n"
        "        <step id=\"2\" name=\"Contextualize (Hypothesize)\">\n"
        "            <context type=\"temporal\">When? (e.g., weekday morning -> commute?)</context>\n"
        "            <context type=\"spatial\">Where? (e.g., office -> work?)</context>\n"
        "            <context type=\"social\">To whom? (e.g., friend, public?)</context>\n"
        "            <context type=\"causal\">Why? (e.g., \"I'm hungry\" -> next is food talk?)</context>\n"
        "            <context type=\"telic\">Goal? (e.g., info-gathering, empathy, decision-making?)</context>\n"
        "        </step>\n"
        "        <step id=\"3\" name=\"Extract Latent Intent\">\n"
        "            <desc>Identify the \"real question\" or \"unspoken need\" behind the literal words.</desc>\n"
        "            <ex>\"Any good cafes around here?\" -> might mean \"Need a quiet place with Wi-Fi to work.\"</ex>\n"
        "        </step>\n"
        "        <step id=\"4\" name=\"Describe Similarity\">\n"
        "            <desc>For each concept, articulate how closely the latent intent aligns with the concept definition and detail, referencing SSR anchors (Core/Strong/Reasonable/Weak/None).</desc>\n"
        "        </step>\n"
        "    </process>\n"
        "\n"
        "    <output>\n"
        "        <primary>Return a JSON object with the single key \"analyses\" whose value is an array of length N. Each element must be a short paragraph (1-2 sentences) written in clear English that begins with one of [Core|Strong|Reasonable|Weak|None] and explains the similarity between the utterance and the concept's definition/detail.</primary>\n"
        "        <ban>Do not output numeric ratings or markdown tables. No additional keys beyond \"analyses\".</ban>\n"
        "    </output>\n"
        "\n"
        "    <example>\n"
        "        <out>{\"analyses\": [\"Strong: Mentions planning a cafe visit matching the concept...\", \"Weak: Only tangential reference...\"]}</out>\n"
        "    </example>\n"
        "</prompt>"
    )
    numeric_system_prompt: str = (
        "<prompt>\n"
        "    <role id=\"Semantic Inference Engine & Latent Intent Decoder\">\n"
        "        <spec>Noisy text (SNS, UGC, transcripts) analysis. Infer user's true goals, emotions, and unstated assumptions from fragmented data.</spec>\n"
        "        <ban>Surface-level keyword matching is strictly forbidden.</ban>\n"
        "    </role>\n"
        "\n"
        "    <mission>For a given utterance and N concepts, calculate a relevance score 'r' (float 0.0-1.0, no rounding) based on the utterance's core intent. Execute via the internal process below.</mission>\n"
        "\n"
        "    <process>\n"
        "        <step id=\"1\" name=\"Normalize & Enrich\">\n"
        "            <task>Normalize slang, typos, jargon.</task>\n"
        "            <task>Interpret non-verbal cues (emojis, punctuation, irony markers) for emotional tone.</task>\n"
        "        </step>\n"
        "        <step id=\"2\" name=\"Contextualize (Hypothesize)\">\n"
        "            <context type=\"temporal\">When? (e.g., weekday morning -> commute?)</context>\n"
        "            <context type=\"spatial\">Where? (e.g., office -> work?)</context>\n"
        "            <context type=\"social\">To whom? (e.g., friend, public?)</context>\n"
        "            <context type=\"causal\">Why? (e.g., \"I'm hungry\" -> next is food talk?)</context>\n"
        "            <context type=\"telic\">Goal? (e.g., info-gathering, empathy, decision-making?)</context>\n"
        "        </step>\n"
        "        <step id=\"3\" name=\"Extract Latent Intent\">\n"
        "            <desc>Identify the \"real question\" or \"unspoken need\" behind the literal words.</desc>\n"
        "            <ex>\"Any good cafes around here?\" -> might mean \"Need a quiet place with Wi-Fi to work.\"</ex>\n"
        "        </step>\n"
        "        <step id=\"4\" name=\"Map & Score\">\n"
        "            <desc>Semantically map the latent intent to each concept's definition. Score relevance 'r' based on the criteria below.</desc>\n"
        "        </step>\n"
        "    </process>\n"
        "\n"
        "    <criteria type=\"relevance_score_r\">\n"
        "        <score r=\"0.9-1.0\" name=\"Core\">Intent and concept are identical. The utterance exists to express the concept.</score>\n"
        "        <score r=\"0.7-0.89\" name=\"Strong\">Concept is the primary subject, strongly inferred from context and intent.</score>\n"
        "        <score r=\"0.4-0.69\" name=\"Reasonable\">Concept is a logical extension or component of the intent.</score>\n"
        "        <score r=\"0.1-0.39\" name=\"Weak\">Faintly associated by situation/words, but not the main focus.</score>\n"
        "        <score r=\"0.0\" name=\"None\">No logical connection can be inferred.</score>\n"
        "    </criteria>\n"
        "\n"
        "    <rules>\n"
        "        <rule id=\"lang\">Auto-detect language, internally translate to a standard model (e.g., English) for processing.</rule>\n"
        "        <rule id=\"silent\">Internal thought processes must NOT be included in the output.</rule>\n"
        "    </rules>\n"
        "\n"
        "    <output>\n"
        "        <primary>Return an N-length array of float numbers (0.0–1.0), ordered by the given concepts. No rounding.</primary>\n"
        "        <compat>If the platform enforces a JSON object wrapper, return only: {\"scores\": [..the same array..]} with no extra keys or text.</compat>\n"
        "        <ban>No extra text, explanations, or markdown.</ban>\n"
        "    </output>\n"
        "\n"
        "    <example>\n"
        "        <out>[0.85, 0.1, 0.65]</out>\n"
        "    </example>\n"
        "</prompt>"
    )
    system_prompt: Optional[str] = None
    spreadsheet_url: str = Field(..., min_length=5)
    sheet_keyword: str = "Link"
    score_sheet_keyword: str = "Embedding"
    spreadsheet_id: Optional[str] = None
    sheet_name: Optional[str] = None
    sheet_gid: Optional[int] = None
    score_sheet_name: Optional[str] = None
    score_sheet_gid: Optional[int] = None
    mode: Literal["csv", "video"] = "csv"
    video_concurrency_default: conint(ge=1, le=50) = 7
    video_timeout_default: conint(ge=10, le=900) = 300
    video_download_timeout: conint(ge=10, le=600) = 120
    video_temp_dir: Optional[str] = None

    @model_validator(mode="after")
    def _normalize_models(self) -> "RunConfig":
        def _default_primary(mode: str) -> str:
            return "gemini-flash-latest" if mode == "video" else "gemini-flash-lite-latest"

        primary = (self.primary_model or "").strip()
        if not primary:
            primary = _default_primary(self.mode)
        self.primary_model = primary

        fallback = (self.fallback_model or "").strip()
        if self.mode == "video":
            self.fallback_model = None
        else:
            self.fallback_model = fallback or "gpt-5-nano"
        return self

    @model_validator(mode="after")
    def _apply_system_prompt(self) -> "RunConfig":
        if not self.system_prompt:
            self.system_prompt = (
                self.ssr_system_prompt if self.enable_ssr else self.numeric_system_prompt
            )
        return self

    @model_validator(mode="after")
    def _validate_slice_bounds(self) -> "RunConfig":
        if (
            self.slice_start_row is not None
            and self.slice_end_row is not None
            and self.slice_end_row < self.slice_start_row
        ):
            raise ValueError("slice_end_row must be greater than or equal to slice_start_row")
        return self

    @property
    def active_system_prompt(self) -> str:
        if self.system_prompt:
            return self.system_prompt
        return self.ssr_system_prompt if self.enable_ssr else self.numeric_system_prompt


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class CreateJobRequest(BaseModel):
    config: RunConfig


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus


class ProgressResponse(BaseModel):
    job_id: str
    status: JobStatus
    total_rows: int
    processed_rows: int
    current_utterance_index: Optional[int] = None
    current_category_block_index: Optional[int] = None
    eta_seconds: Optional[float] = None


@dataclass
class Category:
    name: str
    definition: str
    detail: str


class ScoreRequest(BaseModel):
    utterance: str
    categories: List[Category]
    system_prompt: str
    provider: Provider
    timeout_sec: int = 60
    file_parts: Optional[List[dict]] = None
    model_override: Optional[str] = None
    ssr_enabled: bool = True


class ScoreResult(BaseModel):
    provider: Provider
    model: str
    scores: Optional[List[Optional[confloat(ge=-1.0, le=1.0)]]] = None
    analyses: Optional[List[str]] = None
    likert_pmfs: Optional[List[List[float]]] = None
    raw_text: Optional[str] = None
    request_text: Optional[str] = None
    pre_scores: Optional[List[Optional[float]]] = None
    absolute_scores: Optional[List[Optional[float]]] = None
    relative_rank_scores: Optional[List[Optional[float]]] = None
    anchor_labels: Optional[List[Optional[str]]] = None
    missing_indices: Optional[List[int]] = None
    partial: bool = False


class CleansingJobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class CleansingJobConfig(BaseModel):
    sheet: str = Field(..., min_length=5)
    country: str = Field(..., min_length=1)
    product_category: str = Field(..., min_length=1)
    sheet_name: str = Field(default="RawData_Master", min_length=1)
    concurrency: conint(ge=1, le=200) = 50
    sheet_gid: Optional[int] = None


class CleansingJobResponse(BaseModel):
    job_id: str
    status: CleansingJobStatus


class CleansingRowError(BaseModel):
    row_number: int
    reason: str


class CleansingJobProgress(BaseModel):
    job_id: str
    status: CleansingJobStatus
    total_items: int = 0
    processed_items: int = 0
    success_count: int = 0
    fallback_count: int = 0
    failure_count: int = 0
    message: Optional[str] = None
    errors: Optional[List[CleansingRowError]] = None


class InterviewJobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class InterviewJobConfig(BaseModel):
    project_name: str = Field(..., min_length=3)
    domain: str = Field(..., min_length=2)
    country_region: Optional[str] = None
    stimuli_source: Optional[str] = None
    stimuli_sheet_url: Optional[str] = None
    stimuli_sheet_name: Optional[str] = None
    stimuli_sheet_column: str = Field("A", min_length=1, max_length=3)
    stimuli_sheet_start_row: conint(ge=1) = 2
    persona_sheet_url: Optional[str] = None
    persona_sheet_name: str = "LLMSetUp"
    persona_overview_column: str = Field("B", min_length=1, max_length=3)
    persona_prompt_column: str = Field("C", min_length=1, max_length=3)
    persona_start_row: conint(ge=1) = 2
    persona_count: conint(ge=1, le=500) = 60
    persona_template: Optional[str] = None
    persona_seed: conint(ge=0, le=1_000_000) = 42
    concurrency: conint(ge=1, le=200) = 20
    max_rounds: conint(ge=1, le=1000) = 3
    language: str = "ja"
    language_label: Optional[str] = None
    language_source: Optional[str] = None
    language_reason: Optional[str] = None
    stimulus_mode: Literal["text", "image", "mixed"] = "text"
    notes: Optional[str] = None
    enable_tribe_learning: bool = False
    utterance_csv_path: Optional[str] = None
    manual_stimuli_images: List[str] = Field(default_factory=list)
    tribe_count: conint(ge=1, le=200) = 10
    persona_per_tribe: conint(ge=1, le=50) = 3
    questions_per_persona: conint(ge=1, le=30) = 5
    max_utterance_tokens: conint(ge=100_000, le=2_000_000) = 500_000

    @model_validator(mode="after")
    def _ensure_rounds_match(self) -> "InterviewJobConfig":
        if self.max_rounds != self.questions_per_persona:
            raise ValueError("max_rounds must equal questions_per_persona")
        return self


class InterviewJobResponse(BaseModel):
    job_id: str
    status: InterviewJobStatus


class InterviewJobProgress(BaseModel):
    job_id: str
    status: InterviewJobStatus
    stage: str
    total_personas: int = 0
    generated_personas: int = 0
    processed_transcripts: int = 0
    message: Optional[str] = None
    artifacts: Optional[Dict[str, str]] = None


class TribeInterviewJobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class TribeInterviewStage(str, Enum):
    pending = "pending"
    tribe = "tribe"
    combination = "combination"
    persona = "persona"
    qa = "qa"
    embedding = "embedding"


class TribeInterviewMode(str, Enum):
    product = "product"
    communication = "communication"


class TribeInterviewSheetNames(BaseModel):
    tribe_setup: str = "Tribe_SetUp"
    tribe_combination: str = "Tribe_Combination"
    persona_setup: str = "Persona_SetUp"
    qa_llm: str = "QA_LLM"
    qa_embedding: str = "QA_Embedding"


class TribeInterviewJobConfig(BaseModel):
    product_category: str
    country_region: str
    mode: TribeInterviewMode
    persona_per_combination: conint(ge=1, le=10)
    interviews_per_persona: conint(ge=1, le=10)
    sheet_names: TribeInterviewSheetNames = Field(default_factory=TribeInterviewSheetNames)
    spreadsheet_url: str
    product_detail: Optional[str] = None
    tagline_detail: Optional[str] = None
    image_paths: List[str] = Field(default_factory=list)
    retry_limit: conint(ge=1, le=10) = 5
    max_tribes: conint(ge=1, le=5) = 5
    persona_prompt_template: Optional[str] = None
    interview_questions: Optional[List[str]] = None


class TribeInterviewJobResponse(BaseModel):
    job_id: str
    status: TribeInterviewJobStatus


class TribeInterviewJobProgress(BaseModel):
    job_id: str
    status: TribeInterviewJobStatus
    stage: TribeInterviewStage
    message: Optional[str] = None
    artifacts: Optional[Dict[str, str]] = None
    metrics: Optional[Dict[str, int]] = None




class MassPersonaJobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class MassPersonaJobConfig(BaseModel):
    project_name: str = Field(..., min_length=3)
    domain: str = Field(..., min_length=2)
    language: Literal["ja", "en"] = "ja"
    persona_goal: conint(ge=1, le=5000) = 200
    utterance_source: Optional[str] = None
    default_region: Optional[str] = None
    sheet_url: Optional[str] = None
    sheet_name: Optional[str] = None
    sheet_utterance_column: str = Field("A", min_length=1, max_length=3)
    sheet_region_column: Optional[str] = Field(None, min_length=1, max_length=3)
    sheet_tags_column: Optional[str] = Field(None, min_length=1, max_length=3)
    sheet_start_row: conint(ge=1) = 2
    max_records: conint(ge=1, le=20000) = 2000
    notes: Optional[str] = None
    direction_axes: Dict[str, List[str]] = Field(default_factory=dict)
    direction_must_cover: List[str] = Field(default_factory=list)
    direction_seed_notes: Optional[str] = None


class MassPersonaJobResponse(BaseModel):
    job_id: str
    status: MassPersonaJobStatus


class MassPersonaJobProgress(BaseModel):
    job_id: str
    status: MassPersonaJobStatus
    stage: str
    total_seed_records: int = 0
    processed_records: int = 0
    generated_blueprints: int = 0
    message: Optional[str] = None
    artifacts: Optional[Dict[str, str]] = None


class PersonaBuildJobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class PersonaBuildJobConfig(BaseModel):
    project_name: str = Field(..., min_length=3)
    domain: str = Field(..., min_length=2)
    language: Literal["ja", "en"] = "ja"
    blueprint_path: str = Field(..., min_length=3, description="Mass Persona Seed Ingestで生成したJSONファイルへのパス")
    output_dir: Optional[str] = Field(None, description="runs/ 以下の出力先ディレクトリ")
    persona_goal: conint(ge=1, le=5000) = 200
    concurrency: conint(ge=1, le=50) = 6
    persona_seed_offset: conint(ge=0, le=1_000_000) = 0
    openai_model: str = "gpt-4.1"
    persona_sheet_url: Optional[str] = None
    persona_sheet_name: Optional[str] = None
    persona_overview_column: str = Field("B", min_length=1, max_length=3)
    persona_prompt_column: str = Field("C", min_length=1, max_length=3)
    persona_start_row: conint(ge=1) = 2
    notes: Optional[str] = None


class PersonaBuildJobResponse(BaseModel):
    job_id: str
    status: PersonaBuildJobStatus


class PersonaBuildJobProgress(BaseModel):
    job_id: str
    status: PersonaBuildJobStatus
    stage: str
    total_blueprints: int = 0
    generated_personas: int = 0
    message: Optional[str] = None
    artifacts: Optional[Dict[str, str]] = None


class PersonaResponseJobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class PersonaResponseJobConfig(BaseModel):
    project_name: str = Field(..., min_length=3)
    domain: str = Field(..., min_length=2)
    language: Literal["ja", "en"] = "ja"
    persona_catalog_path: str = Field(..., min_length=3)
    stimuli_source: Optional[str] = None
    stimuli_sheet_url: Optional[str] = None
    stimuli_sheet_name: Optional[str] = None
    stimuli_sheet_column: str = Field("A", min_length=1, max_length=3)
    stimuli_sheet_start_row: conint(ge=1) = 2
    output_dir: Optional[str] = None
    persona_limit: Optional[conint(ge=1, le=5000)] = None
    stimuli_limit: Optional[conint(ge=1, le=500)] = None
    concurrency: conint(ge=1, le=100) = 12
    gemini_model: str = "gemini-flash-latest"
    response_style: Literal["monologue", "qa"] = "monologue"
    include_structured_summary: bool = True
    ssr_reference_path: Optional[str] = None
    ssr_reference_set: Optional[str] = None
    ssr_embeddings_column: str = "embedding"
    ssr_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ssr_device: Optional[str] = None
    ssr_temperature: confloat(ge=0.0) = 1.0
    ssr_epsilon: confloat(ge=0.0) = 0.0
    notes: Optional[str] = None


class PersonaResponseJobResponse(BaseModel):
    job_id: str
    status: PersonaResponseJobStatus


class PersonaResponseJobProgress(BaseModel):
    job_id: str
    status: PersonaResponseJobStatus
    stage: str
    total_pairs: int = 0
    processed_pairs: int = 0
    message: Optional[str] = None
    artifacts: Optional[Dict[str, str]] = None


class PersonaDirectionConfig(BaseModel):
    project_name: str = Field(..., min_length=3)
    domain: str = Field(..., min_length=2)
    language: Literal["ja", "en"] = "ja"
    persona_goal: conint(ge=1, le=5000) = 200
    axes: Dict[str, List[str]] = Field(default_factory=dict)
    must_cover_attributes: List[str] = Field(default_factory=list)
    seed_insights: Optional[str] = None
    notes: Optional[str] = None
    model_name: str = "gemini-pro-latest"
