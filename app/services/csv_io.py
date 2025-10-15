from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from ..models import Category, RunConfig


@dataclass
class ParsedCSV:
    header_rows: List[List[str]]
    data_rows: List[List[str]]
    total_rows: int
    total_cols: int


def _read_csv_bytes(data: bytes) -> List[List[str]]:
    text = data.decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    return [row for row in reader]


def parse_input_csv(data: bytes) -> ParsedCSV:
    rows = _read_csv_bytes(data)
    total_rows = len(rows)
    total_cols = max((len(r) for r in rows), default=0)
    header_rows = rows[:4]
    data_rows = rows[4:]
    return ParsedCSV(header_rows=header_rows, data_rows=data_rows, total_rows=total_rows, total_cols=total_cols)


def read_categories(rows: List[List[str]], cfg: RunConfig, col_offset: int) -> List[Category]:
    cats: List[Category] = []
    # Convert 1-based to 0-based
    name_r = cfg.name_row - 1
    def_r = cfg.def_row - 1
    detail_r = cfg.detail_row - 1
    start_c = cfg.category_start_col - 1 + col_offset
    for i in range(cfg.batch_size):
        c = start_c + i
        name = rows[name_r][c] if name_r < len(rows) and c < len(rows[name_r]) else ""
        definition = rows[def_r][c] if def_r < len(rows) and c < len(rows[def_r]) else ""
        detail = rows[detail_r][c] if detail_r < len(rows) and c < len(rows[detail_r]) else ""
        if name == "" and definition == "" and detail == "":
            # stop early if no more category columns
            break
        cats.append(Category(name=name, definition=definition, detail=detail))
    return cats


def write_output_csv(
    original_bytes: bytes,
    cfg: RunConfig,
    scores_matrix: List[List[Optional[float]]],
    out_path: Path,
) -> None:
    rows = _read_csv_bytes(original_bytes)
    # Apply scores starting from start_row and category_start_col
    for row_idx, scores in enumerate(scores_matrix, start=cfg.start_row - 1):
        if row_idx >= len(rows):
            break
        row = rows[row_idx]
        # Ensure row has enough cols
        needed_len = cfg.category_start_col - 1 + max(len(scores), 0)
        if len(row) < needed_len:
            row.extend([""] * (needed_len - len(row)))
        for i, s in enumerate(scores or []):
            col = cfg.category_start_col - 1 + i
            if s is not None:
                row[col] = f"{s:.2f}"
            # None は上書きしない（レジューム時に既存値を保持）

    # Write CSV
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Zip if >100MB
    if out_path.stat().st_size > 100 * 1024 * 1024:
        import zipfile

        zip_path = out_path.with_suffix(out_path.suffix + ".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(out_path, arcname=out_path.name)
