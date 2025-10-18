from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from datetime import datetime
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
    max_retries: conint(ge=0, le=50) = 10
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
    writer_flush_interval_sec: confloat(ge=0.1, le=30.0) = 2.0
    writer_flush_batch_size: conint(ge=1, le=500) = 50
    writer_retry_limit: conint(ge=1, le=10) = 5
    writer_retry_initial_delay_sec: confloat(ge=0.1, le=10.0) = 0.5
    writer_retry_backoff_multiplier: confloat(ge=1.0, le=5.0) = 2.0

    # Providers
    primary_provider: Provider = Provider.gemini
    fallback_provider: Provider = Provider.openai
    enable_ssr: bool = True
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
    system_prompt: str = (
        "<prompt>"
        "    <role id=\"Semantic Inference Engine & Latent Intent Decoder\">"
        "        <spec>Noisy text (SNS, UGC, transcripts) analysis. Infer user's true goals, emotions, and unstated assumptions from fragmented data.</spec>"
        "        <ban>Surface-level keyword matching is strictly forbidden.</ban>"
        "    </role>"
        "\n"
        "    <mission>For a given utterance and N concepts, perform Semantic Similarity Rating (SSR) as described in AnyAI Scoring research (2025-10-14). First verbalize how similar each concept definition is to the latent intent of the utterance. Then provide succinct natural-language rationales that can be embedded for quantitative scoring.</mission>"
        "\n"
        "    <process>"
        "        <step id=\"1\" name=\"Normalize & Enrich\">"
        "            <task>Normalize slang, typos, jargon.</task>"
        "            <task>Interpret non-verbal cues (emojis, punctuation, irony markers) for emotional tone.</task>"
        "        </step>"
        "        <step id=\"2\" name=\"Contextualize (Hypothesize)\">"
        "            <context type=\"temporal\">When? (e.g., weekday morning -> commute?)</context>"
        "            <context type=\"spatial\">Where? (e.g., office -> work?)</context>"
        "            <context type=\"social\">To whom? (e.g., friend, public?)</context>"
        "            <context type=\"causal\">Why? (e.g., \"I'm hungry\" -> next is food talk?)</context>"
        "            <context type=\"telic\">Goal? (e.g., info-gathering, empathy, decision-making?)</context>"
        "        </step>"
        "        <step id=\"3\" name=\"Extract Latent Intent\">"
        "            <desc>Identify the \"real question\" or \"unspoken need\" behind the literal words.</desc>"
        "            <ex>\"Any good cafes around here?\" -> might mean \"Need a quiet place with Wi-Fi to work.\"</ex>"
        "        </step>"
        "        <step id=\"4\" name=\"Describe Similarity\">"
        "            <desc>For each concept, articulate how closely the latent intent aligns with the concept definition and detail, referencing SSR anchors (Core/Strong/Reasonable/Weak/None).</desc>"
        "        </step>"
        "    </process>"
        "\n"
        "    <output>"
        "        <primary>Return a JSON object with the single key \"analyses\" whose value is an array of length N. Each element must be a short paragraph in natural language (1-2 sentences) that begins with one of [Core|Strong|Reasonable|Weak|None] and explains the similarity between the utterance and the concept's definition/detail.</primary>"
        "        <ban>Do not output numeric ratings or markdown tables. No additional keys beyond \"analyses\".</ban>"
        "    </output>"
        "\n"
        "    <example>"
        "        <out>{\"analyses\": [\"Strong: Mentions planning a cafe visit matching the concept...\", \"Weak: Only tangential reference...\"]}</out>"
        "    </example>"
        "</prompt>"
    )


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


class ScoreResult(BaseModel):
    provider: Provider
    model: str
    scores: Optional[List[Optional[confloat(ge=-1.0, le=1.0)]]] = None
    analyses: Optional[List[str]] = None
    likert_pmfs: Optional[List[List[float]]] = None
    raw_text: Optional[str] = None
    request_text: Optional[str] = None
    pre_scores: Optional[List[Optional[float]]] = None
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
    enable_ssr: bool = False
    max_rounds: conint(ge=1, le=1000) = 3
    language: Literal["ja", "en"] = "ja"
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


class DashboardMetric(BaseModel):
    name: str
    value: str | float
    unit: Optional[str] = None
    delta: Optional[float] = None
    description: Optional[str] = None


class DashboardDataset(BaseModel):
    name: str
    summary: Optional[str] = None
    metrics: Optional[List[DashboardMetric]] = None
    table_headers: Optional[List[str]] = None
    table_rows: Optional[List[List[str | float]]] = None
    chart_type: Optional[str] = None
    provenance: Optional[str] = None


class DashboardRequest(BaseModel):
    title: str
    overview: str
    segments: Optional[List[str]] = None
    highlights: Optional[List[str]] = None
    datasets: List[DashboardDataset] = Field(default_factory=list)
    interview_insights: Optional[List[str]] = None
    scoring_insights: Optional[List[str]] = None
    cleansing_insights: Optional[List[str]] = None
    call_to_actions: Optional[List[str]] = None


class DashboardResponse(BaseModel):
    html: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    plan_markdown: Optional[str] = None
    plan_model: Optional[str] = None
    plan_input_tokens: Optional[int] = None
    plan_output_tokens: Optional[int] = None


class DashboardBuildRequest(BaseModel):
    request: DashboardRequest
    output_dir: str = Field(..., description="相対パス（runs/ 以下）")
    filename: str = Field("index.html", description="出力するHTMLファイル名")
    plan_filename: str = Field("dashboard_plan.md", description="出力する計画ファイル名")


class DashboardFilters(BaseModel):
    segments: Dict[str, List[str]] = Field(default_factory=dict)
    stimuli: List[str] = Field(default_factory=list)
    persona_ids: List[str] = Field(default_factory=list)
    min_ssr_score: Optional[float] = None
    max_ssr_score: Optional[float] = None
    search_text: Optional[str] = None


class DashboardQueryRequest(BaseModel):
    filters: Optional[DashboardFilters] = None
    include_records: bool = True
    limit: Optional[conint(ge=1, le=5000)] = 200


class DashboardFilterOption(BaseModel):
    value: str
    count: int


class DashboardFilterOptions(BaseModel):
    segments: Dict[str, List[DashboardFilterOption]] = Field(default_factory=dict)
    stimuli: List[DashboardFilterOption] = Field(default_factory=list)


class DashboardQueryResponse(BaseModel):
    request: DashboardRequest
    total_responses: int
    filtered_responses: int
    total_personas: int
    filtered_personas: int
    total_stimuli: int
    filtered_stimuli: int
    filters: DashboardFilters = Field(default_factory=DashboardFilters)
    available_filters: DashboardFilterOptions = Field(default_factory=DashboardFilterOptions)
    available_filters_all: DashboardFilterOptions = Field(default_factory=DashboardFilterOptions)
    records: Optional[List[Dict[str, str | float | None]]] = None
    cache_status: Optional[str] = None


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


class DashboardRunSummary(BaseModel):
    job_id: str
    project_name: Optional[str] = None
    domain: Optional[str] = None
    language: Optional[str] = None
    output_dir: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    total_pairs: Optional[int] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)


class DashboardRunDetail(DashboardRunSummary):
    config: PersonaResponseJobConfig


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
