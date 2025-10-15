from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import polars as pl

from ..models import (
    DashboardDataset,
    DashboardFilterOption,
    DashboardFilterOptions,
    DashboardFilters,
    DashboardMetric,
    DashboardRequest,
)


def prepare_dashboard_base(
    personas: List[dict],
    responses: List[dict],
    stimuli: Iterable[str],
    ssr_entries: Optional[List[dict]] = None,
) -> Tuple[pl.DataFrame, int]:
    persona_df = _persona_frame(personas)
    response_df = _response_frame(responses)
    if response_df.is_empty():
        raise ValueError("Persona responses are empty; dashboardデータを生成できません")

    df = response_df.join(persona_df, on="persona_id", how="left")

    if ssr_entries:
        ssr_map = {
            (entry.get("persona_id"), entry.get("stimulus")): _expected_score(entry.get("pmf"))
            for entry in ssr_entries
            if entry.get("pmf")
        }
        df = df.with_columns(
            pl.struct(["persona_id", "stimulus"]).map_elements(lambda s: ssr_map.get((s["persona_id"], s["stimulus"]))).alias("ssr_score")
        )
    else:
        df = df.with_columns(pl.lit(None).alias("ssr_score"))

    expected_stimuli = len(set(stimuli))
    return df, expected_stimuli


def build_dashboard_from_base(
    base_df: pl.DataFrame,
    expected_stimuli: int,
    *,
    filters: Optional[DashboardFilters] = None,
) -> Tuple[DashboardRequest, pl.DataFrame, dict]:
    base_df = base_df.clone()
    base_totals = _compute_totals(base_df, expected_stimuli)

    filtered_df = _apply_filters(base_df, filters) if filters else base_df
    filtered_totals = _compute_totals(filtered_df, expected_stimuli)

    overview = (
        f"{filtered_totals['personas']} personas / {filtered_totals['stimuli']} stimuli / "
        f"{filtered_totals['responses']} responses."
    )
    if filters and filtered_totals["responses"] != base_totals["responses"]:
        overview += f" (総数 {base_totals['responses']})"

    region_dataset = _build_dataset(
        filtered_df,
        group_key="region",
        title="Region Overview",
        chart_type="bar",
    )
    age_dataset = _build_dataset(
        filtered_df,
        group_key="age_band",
        title="Age Band Distribution",
        chart_type="bar",
    )
    attitude_dataset = _build_dataset(
        filtered_df,
        group_key="attitude_cluster",
        title="Attitude Cluster",
        chart_type="bar",
    )
    occupation_dataset = _build_dataset(
        filtered_df,
        group_key="occupation",
        title="Occupation Insight",
        chart_type="bar",
    )
    stimulus_dataset = _build_dataset(
        filtered_df,
        group_key="stimulus",
        title="Stimulus Responses",
        chart_type="column",
        top_n=15,
    )
    heatmap_dataset = _build_heatmap_dataset(filtered_df, title="Age × Stimulus Heatmap")

    ssr_dataset, ssr_summary = _build_ssr_quantile_dataset(filtered_df)

    highlights = _build_highlights(
        region_dataset,
        age_dataset,
        attitude_dataset,
        occupation_dataset,
        stimulus_dataset,
        ssr_dataset,
    )

    request = DashboardRequest(
        title="Persona Insight Dashboard v1.0",
        overview=overview,
        segments=["region", "age_band", "attitude_cluster", "occupation"],
        highlights=highlights,
        datasets=[
            region_dataset,
            age_dataset,
            attitude_dataset,
            occupation_dataset,
            stimulus_dataset,
            heatmap_dataset,
            *( [ssr_dataset] if ssr_dataset else [] ),
        ],
        interview_insights=None,
    )

    meta = {
        "total_responses": base_totals["responses"],
        "filtered_responses": filtered_totals["responses"],
        "total_personas": base_totals["personas"],
        "filtered_personas": filtered_totals["personas"],
        "total_stimuli": base_totals["stimuli"],
        "filtered_stimuli": filtered_totals["stimuli"],
        "filter_options_all": _collect_filter_options(base_df),
        "filter_options_filtered": _collect_filter_options(filtered_df),
        "ssr_summary": ssr_summary,
        "expected_stimuli": expected_stimuli,
    }

    return request, filtered_df, meta


def build_persona_insight_request(
    personas: List[dict],
    responses: List[dict],
    stimuli: Iterable[str],
    ssr_entries: Optional[List[dict]] = None,
    filters: Optional[DashboardFilters] = None,
) -> Tuple[DashboardRequest, pl.DataFrame, dict]:
    base_df, expected_stimuli = prepare_dashboard_base(personas, responses, stimuli, ssr_entries)
    return build_dashboard_from_base(base_df, expected_stimuli, filters=filters)


def _persona_frame(personas: List[dict]) -> pl.DataFrame:
    rows = []
    for persona in personas:
        persona_type = persona.get("persona_type") or {}
        rows.append(
            {
                "persona_id": persona.get("persona_id"),
                "age_band": persona.get("age_band") or persona_type.get("age_band"),
                "income_band": persona.get("income_band") or persona_type.get("income_band"),
                "region": persona.get("region") or persona_type.get("region"),
                "attitude_cluster": persona.get("attitude_cluster") or persona_type.get("attitude_cluster"),
                "occupation": persona.get("occupation") or persona_type.get("occupation"),
            }
        )
    if not rows:
        rows.append({
            "persona_id": None,
            "age_band": None,
            "income_band": None,
            "region": None,
            "attitude_cluster": None,
            "occupation": None,
        })
    return pl.DataFrame(rows)


def _response_frame(responses: List[dict]) -> pl.DataFrame:
    rows = []
    for record in responses:
        structured = record.get("structured") or {}
        rows.append(
            {
                "persona_id": record.get("persona_id"),
                "stimulus": record.get("stimulus"),
                "response": record.get("response"),
                "purchase_intent": structured.get("purchase_intent"),
                "strength_of_intent": structured.get("strength_of_intent"),
            }
        )
    return pl.DataFrame(rows)


def _apply_filters(df: pl.DataFrame, filters: Optional[DashboardFilters]) -> pl.DataFrame:
    if not filters:
        return df

    df_aug = df
    norm_cols: set[str] = set()
    conditions: List[pl.Expr] = []

    segment_filters = filters.segments or {}
    for column, values in segment_filters.items():
        if not values or column not in df_aug.columns:
            continue
        norm_col = f"__norm_{column}"
        if norm_col not in df_aug.columns:
            df_aug = df_aug.with_columns(
                pl.col(column).map_elements(_normalize_value, return_dtype=pl.Utf8, skip_nulls=False).alias(norm_col)
            )
            norm_cols.add(norm_col)
        normalized_values = [_normalize_filter_label(value) for value in values]
        conditions.append(pl.col(norm_col).is_in(normalized_values))

    if filters.stimuli and "stimulus" in df_aug.columns:
        norm_col = "__norm_stimulus"
        if norm_col not in df_aug.columns:
            df_aug = df_aug.with_columns(
                pl.col("stimulus").map_elements(_normalize_value, return_dtype=pl.Utf8, skip_nulls=False).alias(norm_col)
            )
            norm_cols.add(norm_col)
        normalized_values = [_normalize_filter_label(value) for value in filters.stimuli]
        conditions.append(pl.col(norm_col).is_in(normalized_values))

    if filters.persona_ids and "persona_id" in df_aug.columns:
        normalized_values = [_normalize_filter_label(value) for value in filters.persona_ids]
        conditions.append(pl.col("persona_id").cast(pl.Utf8).is_in(normalized_values))

    if filters.min_ssr_score is not None and "ssr_score" in df_aug.columns:
        conditions.append(pl.col("ssr_score").is_not_null() & (pl.col("ssr_score") >= float(filters.min_ssr_score)))
    if filters.max_ssr_score is not None and "ssr_score" in df_aug.columns:
        conditions.append(pl.col("ssr_score").is_not_null() & (pl.col("ssr_score") <= float(filters.max_ssr_score)))

    if filters.search_text:
        text = filters.search_text.strip().lower()
        if text:
            response_expr = (
                pl.col("response")
                .fill_null("")
                .cast(pl.Utf8)
                .str.to_lowercase()
                .str.contains(text, literal=True)
            )
            stimulus_expr = (
                pl.col("stimulus")
                .fill_null("")
                .cast(pl.Utf8)
                .str.to_lowercase()
                .str.contains(text, literal=True)
            )
            conditions.append(response_expr | stimulus_expr)

    if not conditions:
        if norm_cols:
            return df_aug.drop(list(norm_cols))
        return df_aug

    combined = conditions[0]
    for expr in conditions[1:]:
        combined = combined & expr

    result = df_aug.filter(combined)
    if norm_cols:
        result = result.drop(list(norm_cols))
    return result


def _compute_totals(df: pl.DataFrame, expected_stimuli: int) -> Dict[str, int]:
    responses = int(df.height)
    if "persona_id" in df.columns:
        personas = int(df.get_column("persona_id").n_unique())
    else:
        personas = 0
    if "stimulus" in df.columns:
        stimuli = int(df.get_column("stimulus").n_unique())
    else:
        stimuli = 0
    if expected_stimuli:
        stimuli = max(stimuli, int(expected_stimuli))
    return {"responses": responses, "personas": personas, "stimuli": stimuli}


def _collect_filter_options(df: pl.DataFrame) -> DashboardFilterOptions:
    options = DashboardFilterOptions()
    for column in ("region", "age_band", "attitude_cluster", "occupation"):
        if column in df.columns:
            options.segments[column] = _value_count_options(df, column)
    if "stimulus" in df.columns:
        options.stimuli = _value_count_options(df, "stimulus")
    return options


def _value_count_options(df: pl.DataFrame, column: str) -> List[DashboardFilterOption]:
    counts: Dict[str, int] = {}
    series = df.get_column(column).to_list() if column in df.columns else []
    for value in series:
        normalized = _normalize_value(value)
        counts[normalized] = counts.get(normalized, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [DashboardFilterOption(value=label, count=count) for label, count in sorted_items]


def _normalize_filter_label(value: str | None) -> str:
    return _normalize_value(value)


def _normalize_value(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or "unknown"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return "unknown"
        return str(value)
    return str(value)


def _build_ssr_quantile_dataset(df: pl.DataFrame) -> Tuple[Optional[DashboardDataset], Optional[dict]]:
    if "ssr_score" not in df.columns:
        return None, None
    series = df.get_column("ssr_score").drop_nans().drop_nulls()
    if series.is_empty():
        return None, None

    quantiles = {
        "P10": float(series.quantile(0.10, interpolation="nearest")),
        "P25": float(series.quantile(0.25, interpolation="nearest")),
        "Median": float(series.quantile(0.50, interpolation="nearest")),
        "P75": float(series.quantile(0.75, interpolation="nearest")),
        "P90": float(series.quantile(0.90, interpolation="nearest")),
    }
    mean = float(series.mean())
    std = float(series.std())
    count = int(series.len())
    se = std / math.sqrt(count) if count > 1 else 0.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se

    table_rows = [[label, _format_score(value)] for label, value in quantiles.items()]
    table_rows.extend([
        ["Mean", _format_score(mean)],
        ["Std Dev", _format_score(std)],
        ["Sample Size", str(count)],
        ["95% CI Low", _format_score(ci_low)],
        ["95% CI High", _format_score(ci_high)],
    ])

    dataset = DashboardDataset(
        name="SSR Distribution Summary",
        summary=None,
        metrics=[
            DashboardMetric(
                name="Mean SSR",
                value=_format_score(mean),
                description=f"n = {count}",
            ),
            DashboardMetric(
                name="95% CI",
                value=f"{_format_score(ci_low)} – {_format_score(ci_high)}",
                description="Approximate",
            ),
        ],
        table_headers=["stat", "value"],
        table_rows=table_rows,
        chart_type=None,
    )

    summary = {
        "mean": _format_score(mean),
        "std": _format_score(std),
        "count": count,
        "ci_low": _format_score(ci_low),
        "ci_high": _format_score(ci_high),
        "quantiles": {key: _format_score(value) for key, value in quantiles.items()},
    }

    return dataset, summary


def _build_dataset(
    df: pl.DataFrame,
    *,
    group_key: str,
    title: str,
    chart_type: str,
    top_n: Optional[int] = None,
) -> DashboardDataset:
    if group_key not in df.columns:
        df = df.with_columns(pl.lit("unknown").alias(group_key))

    agg_df = (
        df.group_by(group_key)
        .agg(
            pl.len().alias("responses"),
            pl.col("ssr_score").mean().alias("avg_ssr"),
        )
        .sort("responses", descending=True)
    )
    if top_n:
        agg_df = agg_df.head(top_n)

    table_rows = []
    for row in agg_df.iter_rows(named=True):
        avg_ssr = row.get("avg_ssr")
        table_rows.append([
            row.get(group_key) or "unknown",
            int(row.get("responses") or 0),
            _format_score(avg_ssr),
        ])

    metrics: List[DashboardMetric] = []
    if not agg_df.is_empty():
        top_row = agg_df.row(0, named=True)
        metrics.append(
            DashboardMetric(
                name=f"Top {group_key}",
                value=top_row.get(group_key) or "unknown",
                description=f"Responses: {top_row.get('responses')}",
            )
        )
        if top_row.get("avg_ssr") is not None and not math.isnan(top_row.get("avg_ssr")):
            metrics.append(
                DashboardMetric(
                    name="Avg SSR",
                    value=_format_score(top_row.get("avg_ssr")),
                    description="Likert 推定値",
                )
            )

    return DashboardDataset(
        name=title,
        summary=None,
        metrics=metrics,
        table_headers=[group_key, "responses", "avg_ssr"],
        table_rows=table_rows,
        chart_type=chart_type,
    )


def _build_heatmap_dataset(df: pl.DataFrame, *, title: str) -> DashboardDataset:
    if "age_band" not in df.columns:
        df = df.with_columns(pl.lit("unknown").alias("age_band"))
    pivot_df = (
        df.group_by(["age_band", "stimulus"])
        .agg(pl.col("ssr_score").mean().alias("avg_ssr"))
        .pivot(values="avg_ssr", index="age_band", columns="stimulus")
        .sort("age_band")
    )
    if "age_band" not in pivot_df.columns:
        pivot_df = pivot_df.with_columns(pl.lit("unknown").alias("age_band"))
    table_headers = ["age_band"] + [col for col in pivot_df.columns if col != "age_band"]
    table_rows = []
    for row in pivot_df.iter_rows(named=True):
        table_rows.append([
            row.get("age_band")
        ] + [_format_score(row.get(col)) if isinstance(row.get(col), (int, float)) else row.get(col) for col in table_headers[1:]])

    return DashboardDataset(
        name=title,
        summary=None,
        metrics=None,
        table_headers=table_headers,
        table_rows=table_rows,
        chart_type="heatmap",
    )


def _build_highlights(*datasets: Optional[DashboardDataset]) -> List[str]:
    highlights: List[str] = []
    for dataset in datasets:
        if not dataset:
            continue
        if not dataset.metrics:
            continue
        metric = dataset.metrics[0]
        value = metric.value
        if isinstance(value, (int, float)):
            highlights.append(f"{dataset.name}: {metric.name} = {value}")
        else:
            highlights.append(f"{dataset.name}: {value} ({metric.description})")
    return highlights[:5]


def _expected_score(pmf: Optional[List[float]]) -> Optional[float]:
    if not pmf:
        return None
    weights = list(range(1, len(pmf) + 1))
    total = 0.0
    for w, p in zip(weights, pmf):
        if p is None:
            continue
        total += w * float(p)
    return total


def _format_score(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if math.isnan(value):
        return None
    return round(float(value), 2)
