from __future__ import annotations

from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .google_sheets import column_index_to_a1


def _quote_sheet_name(name: str) -> str:
    """Escape a sheet name for inclusion in an A1 range."""

    return name.replace("'", "''")


def _group_contiguous(offsets: Sequence[int]) -> Iterator[Sequence[int]]:
    if not offsets:
        return

    group: List[int] = [offsets[0]]
    for offset in offsets[1:]:
        if offset == group[-1] + 1:
            group.append(offset)
        else:
            yield tuple(group)
            group = [offset]
    yield tuple(group)


def build_row_value_ranges(
    *,
    category_start_col: int,
    sheet_name: str,
    update_buffer: Mapping[int, Mapping[int, float]],
) -> List[Tuple[int, dict]]:
    """Convert buffered sheet updates into value ranges grouped by row."""

    if not update_buffer:
        return []

    base_col_index = max(0, int(category_start_col) - 1)
    sheet_label = _quote_sheet_name(sheet_name)
    entries: List[Tuple[int, dict]] = []

    for row_number, columns in sorted(update_buffer.items()):
        if not columns:
            continue
        offsets = sorted(columns.keys())
        if not offsets:
            continue
        for group in _group_contiguous(offsets):
            start_offset = group[0]
            end_offset = group[-1]
            start_col = column_index_to_a1(base_col_index + start_offset)
            end_col = column_index_to_a1(base_col_index + end_offset)
            range_label = f"'{sheet_label}'!{start_col}{row_number}:{end_col}{row_number}"
            values = [[columns[offset] for offset in group]]
            entries.append((row_number, {"range": range_label, "values": values}))

    return entries


def chunk_value_ranges_by_row(
    entries: Iterable[Tuple[int, dict]],
    *,
    max_rows_per_batch: Optional[int],
) -> List[List[dict]]:
    """Chunk value ranges so that each batch touches at most the given number of rows."""

    entries_list = list(entries)
    if not entries_list:
        return []

    limit = max_rows_per_batch if max_rows_per_batch and max_rows_per_batch > 0 else None
    batches: List[List[dict]] = []
    current_batch: List[dict] = []
    current_rows: set[int] = set()

    for row_number, payload in entries_list:
        if limit is not None and row_number not in current_rows and len(current_rows) >= limit:
            if current_batch:
                batches.append(current_batch)
            current_batch = []
            current_rows = set()
        current_batch.append(payload)
        current_rows.add(row_number)

    if current_batch:
        batches.append(current_batch)

    return batches


def build_batched_value_ranges(
    *,
    category_start_col: int,
    sheet_name: str,
    update_buffer: Mapping[int, Mapping[int, float]],
    max_rows_per_batch: Optional[int],
) -> List[List[dict]]:
    """Convenience helper returning chunked sheet update payloads."""

    row_entries = build_row_value_ranges(
        category_start_col=category_start_col,
        sheet_name=sheet_name,
        update_buffer=update_buffer,
    )
    return chunk_value_ranges_by_row(row_entries, max_rows_per_batch=max_rows_per_batch)

