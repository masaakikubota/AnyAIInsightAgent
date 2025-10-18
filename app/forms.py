from __future__ import annotations

from typing import List, Optional

from fastapi import UploadFile
from pydantic import BaseModel, field_validator, model_validator

from .models import (
    InterviewJobConfig,
    MassPersonaJobConfig,
    PersonaBuildJobConfig,
    PersonaResponseJobConfig,
)


class _BaseForm(BaseModel):
    @staticmethod
    def _strip_or_none(value: object | None) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _sanitize_column(value: object | None, default: str) -> str:
        candidate = _BaseForm._strip_or_none(value) or default
        upper = candidate.upper()
        return upper if upper.isalpha() else default.upper()

    @staticmethod
    def _coerce_int(
        value: object | None,
        default: int,
        *,
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
    ) -> int:
        try:
            parsed = int(value) if value is not None else default
        except (TypeError, ValueError):
            parsed = default
        if minimum is not None and parsed < minimum:
            parsed = minimum
        if maximum is not None and parsed > maximum:
            parsed = maximum
        return parsed

    @staticmethod
    def _coerce_float(value: object | None, default: float, *, minimum: float = 0.0) -> float:
        try:
            parsed = float(value) if value is not None else default
        except (TypeError, ValueError):
            parsed = default
        return parsed if parsed >= minimum else minimum

    @staticmethod
    def _to_bool(value: object, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "on", "yes"}

    @staticmethod
    def _normalize_language(value: object | None, default: str = "ja") -> str:
        candidate = (_BaseForm._strip_or_none(value) or default).lower()
        return candidate if candidate in {"ja", "en"} else default


class InterviewJobForm(_BaseForm):
    project_name: str
    domain: str
    country_region: Optional[str] = None
    stimuli_source: Optional[str] = None
    stimuli_sheet_url: Optional[str] = None
    stimuli_sheet_name: Optional[str] = None
    stimuli_sheet_column: str = "A"
    stimuli_sheet_start_row: int = 2
    persona_sheet_url: Optional[str] = None
    persona_sheet_name: str = "LLMSetUp"
    persona_overview_column: str = "B"
    persona_prompt_column: str = "C"
    persona_start_row: int = 2
    persona_count: int = 60
    persona_template: Optional[str] = None
    persona_seed: int = 42
    concurrency: int = 20
    enable_ssr: bool = False
    max_rounds: int = 3
    language: str = "ja"
    stimulus_mode: str = "text"
    notes: Optional[str] = None
    enable_tribe_learning: bool = False
    utterance_csv: UploadFile | None = None
    manual_stimuli_images: Optional[List[UploadFile]] = None
    tribe_count: int = 10
    persona_per_tribe: int = 3
    questions_per_persona: int = 5

    @field_validator("project_name", "domain", mode="before")
    def _strip_required(cls, value: object) -> str:
        text = _BaseForm._strip_or_none(value)
        if not text:
            raise ValueError("must not be empty")
        return text

    @field_validator("country_region", mode="before")
    def _normalize_region(cls, value: object | None) -> Optional[str]:
        text = _BaseForm._strip_or_none(value)
        if not text:
            return None
        return "_".join(text.split())

    @field_validator(
        "stimuli_source",
        "stimuli_sheet_url",
        "stimuli_sheet_name",
        "persona_sheet_url",
        "persona_template",
        "notes",
        mode="before",
    )
    def _strip_optional(cls, value: object | None) -> Optional[str]:
        return _BaseForm._strip_or_none(value)

    @field_validator("persona_sheet_name", mode="before")
    def _normalize_persona_sheet_name(cls, value: object | None) -> str:
        return _BaseForm._strip_or_none(value) or "LLMSetUp"

    @field_validator(
        "stimuli_sheet_column",
        "persona_overview_column",
        "persona_prompt_column",
        mode="before",
    )
    def _normalize_columns(cls, value: object, info) -> str:
        default_map = {
            "stimuli_sheet_column": "A",
            "persona_overview_column": "B",
            "persona_prompt_column": "C",
        }
        default = default_map.get(info.field_name, "A")
        return _BaseForm._sanitize_column(value, default)

    @field_validator(
        "stimuli_sheet_start_row",
        "persona_start_row",
        "tribe_count",
        "persona_per_tribe",
        "questions_per_persona",
        "max_rounds",
        "persona_seed",
        "concurrency",
        "persona_count",
        mode="before",
    )
    def _normalize_ints(cls, value: object, info) -> int:
        constraints: dict[str, tuple[int, Optional[int]]] = {
            "stimuli_sheet_start_row": (2, None),
            "persona_start_row": (2, None),
            "tribe_count": (10, 200),
            "persona_per_tribe": (3, 50),
            "questions_per_persona": (5, 30),
            "max_rounds": (3, 30),
            "persona_seed": (42, 1_000_000),
            "concurrency": (20, 200),
            "persona_count": (60, 500),
        }
        default, maximum = constraints.get(info.field_name, (0, None))
        minimum = 1 if info.field_name not in {"persona_seed"} else 0
        return _BaseForm._coerce_int(value, default, minimum=minimum, maximum=maximum)

    @field_validator("language", mode="before")
    def _normalize_language_field(cls, value: object | None) -> str:
        return _BaseForm._normalize_language(value)

    @field_validator("stimulus_mode", mode="before")
    def _normalize_mode(cls, value: object | None) -> str:
        candidate = (_BaseForm._strip_or_none(value) or "text").lower()
        return candidate if candidate in {"text", "image", "mixed"} else "text"

    @field_validator("enable_tribe_learning", mode="before")
    def _normalize_bool(cls, value: object | None) -> bool:
        return _BaseForm._to_bool(value, default=False)

    @model_validator(mode="after")
    def _validate_rounds(self) -> "InterviewJobForm":
        if self.max_rounds != self.questions_per_persona:
            raise ValueError("max_rounds must equal questions_per_persona")
        return self

    @property
    def manual_mode(self) -> bool:
        return not self.stimuli_sheet_url

    def to_config(
        self,
        *,
        utterance_csv_path: Optional[str] = None,
        manual_image_paths: Optional[List[str]] = None,
    ) -> InterviewJobConfig:
        manual_image_paths = manual_image_paths or []
        total_personas = max(1, self.tribe_count * self.persona_per_tribe)
        persona_count = min(total_personas, 500)
        return InterviewJobConfig(
            project_name=self.project_name,
            domain=self.domain,
            country_region=self.country_region,
            stimuli_source=self.stimuli_source,
            stimuli_sheet_url=self.stimuli_sheet_url,
            stimuli_sheet_name=self.stimuli_sheet_name,
            stimuli_sheet_column=self.stimuli_sheet_column,
            stimuli_sheet_start_row=self.stimuli_sheet_start_row,
            persona_sheet_url=self.persona_sheet_url,
            persona_sheet_name=self.persona_sheet_name,
            persona_overview_column=self.persona_overview_column,
            persona_prompt_column=self.persona_prompt_column,
            persona_start_row=self.persona_start_row,
            persona_count=persona_count,
            persona_template=self.persona_template,
            persona_seed=self.persona_seed,
            concurrency=self.concurrency,
            enable_ssr=False,
            max_rounds=self.questions_per_persona,
            language=self.language,
            stimulus_mode=self.stimulus_mode,
            notes=self.notes,
            enable_tribe_learning=self.enable_tribe_learning,
            utterance_csv_path=utterance_csv_path,
            manual_stimuli_images=manual_image_paths,
            tribe_count=self.tribe_count,
            persona_per_tribe=self.persona_per_tribe,
            questions_per_persona=self.questions_per_persona,
            max_utterance_tokens=InterviewJobConfig.model_fields["max_utterance_tokens"].default,
        )


class MassPersonaJobForm(_BaseForm):
    project_name: str
    domain: str
    language: str = "ja"
    persona_goal: int = 200
    utterance_source: Optional[str] = None
    default_region: Optional[str] = None
    sheet_url: Optional[str] = None
    sheet_name: Optional[str] = None
    sheet_utterance_column: str = "A"
    sheet_region_column: Optional[str] = None
    sheet_tags_column: Optional[str] = None
    sheet_start_row: int = 2
    max_records: int = 2000
    notes: Optional[str] = None

    @field_validator("project_name", "domain", mode="before")
    def _strip_required(cls, value: object) -> str:
        text = _BaseForm._strip_or_none(value)
        if not text:
            raise ValueError("must not be empty")
        return text

    @field_validator("language", mode="before")
    def _normalize_language_field(cls, value: object | None) -> str:
        return _BaseForm._normalize_language(value)

    @field_validator(
        "utterance_source",
        "default_region",
        "sheet_url",
        "sheet_name",
        "sheet_region_column",
        "sheet_tags_column",
        "notes",
        mode="before",
    )
    def _strip_optional(cls, value: object | None) -> Optional[str]:
        return _BaseForm._strip_or_none(value)

    @field_validator("sheet_utterance_column", mode="before")
    def _sanitize_sheet_column(cls, value: object | None) -> str:
        return _BaseForm._sanitize_column(value, "A")

    @field_validator(
        "sheet_region_column",
        "sheet_tags_column",
        mode="after",
    )
    def _uppercase_optional_columns(cls, value: Optional[str]) -> Optional[str]:
        return value.upper() if value else None

    @field_validator("persona_goal", mode="before")
    def _clamp_persona_goal(cls, value: object | None) -> int:
        return _BaseForm._coerce_int(value, 200, minimum=1, maximum=5000)

    @field_validator("sheet_start_row", mode="before")
    def _normalize_start_row(cls, value: object | None) -> int:
        return _BaseForm._coerce_int(value, 2, minimum=1)

    @field_validator("max_records", mode="before")
    def _normalize_max_records(cls, value: object | None) -> int:
        return _BaseForm._coerce_int(value, 2000, minimum=1, maximum=20000)

    def to_config(self) -> MassPersonaJobConfig:
        return MassPersonaJobConfig(
            project_name=self.project_name,
            domain=self.domain,
            language=self.language,
            persona_goal=self.persona_goal,
            utterance_source=self.utterance_source,
            default_region=self.default_region,
            sheet_url=self.sheet_url,
            sheet_name=self.sheet_name,
            sheet_utterance_column=self.sheet_utterance_column,
            sheet_region_column=self.sheet_region_column,
            sheet_tags_column=self.sheet_tags_column,
            sheet_start_row=self.sheet_start_row,
            max_records=self.max_records,
            notes=self.notes,
        )


class PersonaBuildJobForm(_BaseForm):
    project_name: str
    domain: str
    language: str = "ja"
    blueprint_path: str
    output_dir: Optional[str] = None
    persona_goal: int = 200
    concurrency: int = 6
    persona_seed_offset: int = 0
    openai_model: str = "gpt-4.1"
    persona_sheet_url: Optional[str] = None
    persona_sheet_name: str = "PersonaCatalog"
    persona_overview_column: str = "B"
    persona_prompt_column: str = "C"
    persona_start_row: int = 2
    notes: Optional[str] = None

    @field_validator("project_name", "domain", "blueprint_path", mode="before")
    def _strip_required(cls, value: object) -> str:
        text = _BaseForm._strip_or_none(value)
        if not text:
            raise ValueError("must not be empty")
        return text

    @field_validator("language", mode="before")
    def _normalize_language_field(cls, value: object | None) -> str:
        return _BaseForm._normalize_language(value)

    @field_validator("output_dir", "persona_sheet_url", "notes", mode="before")
    def _strip_optional(cls, value: object | None) -> Optional[str]:
        return _BaseForm._strip_or_none(value)

    @field_validator("persona_sheet_name", mode="before")
    def _normalize_sheet_name(cls, value: object | None) -> str:
        return _BaseForm._strip_or_none(value) or "PersonaCatalog"

    @field_validator(
        "persona_overview_column",
        "persona_prompt_column",
        mode="before",
    )
    def _sanitize_columns(cls, value: object, info) -> str:
        defaults = {
            "persona_overview_column": "B",
            "persona_prompt_column": "C",
        }
        return _BaseForm._sanitize_column(value, defaults.get(info.field_name, "B"))

    @field_validator("persona_goal", mode="before")
    def _normalize_persona_goal(cls, value: object | None) -> int:
        return _BaseForm._coerce_int(value, 200, minimum=1, maximum=5000)

    @field_validator("concurrency", mode="before")
    def _normalize_concurrency(cls, value: object | None) -> int:
        return _BaseForm._coerce_int(value, 6, minimum=1, maximum=50)

    @field_validator("persona_seed_offset", mode="before")
    def _normalize_seed_offset(cls, value: object | None) -> int:
        return _BaseForm._coerce_int(value, 0, minimum=0, maximum=1_000_000)

    @field_validator("persona_start_row", mode="before")
    def _normalize_start_row(cls, value: object | None) -> int:
        return _BaseForm._coerce_int(value, 2, minimum=1, maximum=1_000_000)

    @field_validator("openai_model", mode="before")
    def _strip_model(cls, value: object | None) -> str:
        return _BaseForm._strip_or_none(value) or "gpt-4.1"

    def to_config(self, *, output_dir: Optional[str] = None) -> PersonaBuildJobConfig:
        return PersonaBuildJobConfig(
            project_name=self.project_name,
            domain=self.domain,
            language=self.language,
            blueprint_path=self.blueprint_path,
            output_dir=output_dir,
            persona_goal=self.persona_goal,
            concurrency=self.concurrency,
            persona_seed_offset=self.persona_seed_offset,
            openai_model=self.openai_model,
            persona_sheet_url=self.persona_sheet_url,
            persona_sheet_name=self.persona_sheet_name,
            persona_overview_column=self.persona_overview_column,
            persona_prompt_column=self.persona_prompt_column,
            persona_start_row=self.persona_start_row,
            notes=self.notes,
        )


class PersonaResponseJobForm(_BaseForm):
    project_name: str
    domain: str
    language: str = "ja"
    persona_catalog_path: str
    stimuli_source: Optional[str] = None
    stimuli_sheet_url: Optional[str] = None
    stimuli_sheet_name: Optional[str] = None
    stimuli_sheet_column: str = "A"
    stimuli_sheet_start_row: int = 2
    output_dir: Optional[str] = None
    persona_limit: int = 0
    stimuli_limit: int = 0
    concurrency: int = 12
    gemini_model: str = "gemini-flash-latest"
    response_style: str = "monologue"
    include_structured_summary: bool = True
    ssr_reference_path: Optional[str] = None
    ssr_reference_set: Optional[str] = None
    ssr_embeddings_column: str = "embedding"
    ssr_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ssr_device: Optional[str] = None
    ssr_temperature: float = 1.0
    ssr_epsilon: float = 0.0
    notes: Optional[str] = None

    @field_validator("project_name", "domain", "persona_catalog_path", mode="before")
    def _strip_required(cls, value: object) -> str:
        text = _BaseForm._strip_or_none(value)
        if not text:
            raise ValueError("must not be empty")
        return text

    @field_validator("language", mode="before")
    def _normalize_language_field(cls, value: object | None) -> str:
        return _BaseForm._normalize_language(value)

    @field_validator(
        "stimuli_source",
        "stimuli_sheet_url",
        "stimuli_sheet_name",
        "output_dir",
        "gemini_model",
        "ssr_reference_path",
        "ssr_reference_set",
        "ssr_device",
        "notes",
        mode="before",
    )
    def _strip_optional(cls, value: object | None) -> Optional[str]:
        return _BaseForm._strip_or_none(value)

    @field_validator("stimuli_sheet_column", mode="before")
    def _sanitize_stimuli_column(cls, value: object | None) -> str:
        return _BaseForm._sanitize_column(value, "A")

    @field_validator("stimuli_sheet_start_row", mode="before")
    def _normalize_start_row(cls, value: object | None) -> int:
        return _BaseForm._coerce_int(value, 2, minimum=1)

    @field_validator("persona_limit", "stimuli_limit", mode="before")
    def _normalize_optional_limits(cls, value: object | None, info) -> int:
        defaults = {"persona_limit": 0, "stimuli_limit": 0}
        parsed = _BaseForm._coerce_int(value, defaults[info.field_name], minimum=0)
        return parsed

    @field_validator("concurrency", mode="before")
    def _normalize_concurrency(cls, value: object | None) -> int:
        return _BaseForm._coerce_int(value, 12, minimum=1, maximum=100)

    @field_validator("response_style", mode="before")
    def _normalize_response_style(cls, value: object | None) -> str:
        candidate = (_BaseForm._strip_or_none(value) or "monologue").lower()
        return "qa" if candidate == "qa" else "monologue"

    @field_validator("include_structured_summary", mode="before")
    def _normalize_include_summary(cls, value: object | None) -> bool:
        return _BaseForm._to_bool(value, default=True)

    @field_validator("ssr_embeddings_column", "ssr_model_name", mode="before")
    def _strip_required_defaults(cls, value: object | None, info) -> str:
        defaults = {
            "ssr_embeddings_column": "embedding",
            "ssr_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }
        return _BaseForm._strip_or_none(value) or defaults[info.field_name]

    @field_validator("ssr_temperature", mode="before")
    def _normalize_temperature(cls, value: object | None) -> float:
        return _BaseForm._coerce_float(value, 1.0, minimum=0.0)

    @field_validator("ssr_epsilon", mode="before")
    def _normalize_epsilon(cls, value: object | None) -> float:
        return _BaseForm._coerce_float(value, 0.0, minimum=0.0)

    def to_config(self, *, output_dir: Optional[str] = None) -> PersonaResponseJobConfig:
        persona_limit = None
        if self.persona_limit > 0:
            persona_limit = min(self.persona_limit, 5000)
        stimuli_limit = None
        if self.stimuli_limit > 0:
            stimuli_limit = min(self.stimuli_limit, 500)
        return PersonaResponseJobConfig(
            project_name=self.project_name,
            domain=self.domain,
            language=self.language,
            persona_catalog_path=self.persona_catalog_path,
            stimuli_source=self.stimuli_source,
            stimuli_sheet_url=self.stimuli_sheet_url,
            stimuli_sheet_name=self.stimuli_sheet_name,
            stimuli_sheet_column=self.stimuli_sheet_column,
            stimuli_sheet_start_row=self.stimuli_sheet_start_row,
            output_dir=output_dir,
            persona_limit=persona_limit,
            stimuli_limit=stimuli_limit,
            concurrency=self.concurrency,
            gemini_model=self.gemini_model or "gemini-flash-latest",
            response_style=self.response_style,
            include_structured_summary=self.include_structured_summary,
            ssr_reference_path=self.ssr_reference_path,
            ssr_reference_set=self.ssr_reference_set,
            ssr_embeddings_column=self.ssr_embeddings_column,
            ssr_model_name=self.ssr_model_name,
            ssr_device=self.ssr_device,
            ssr_temperature=self.ssr_temperature,
            ssr_epsilon=self.ssr_epsilon,
            notes=self.notes,
        )
