#!/usr/bin/env python3
"""
Quick dashboard cache exerciser.

Usage:
    python scripts/test_dashboard_cache.py --job-id <job_id> [--base-dir runs] [--limit 200]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.persona_response_manager import PersonaResponseJobManager
from app.models import DashboardFilters


def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard cache smoke test")
    parser.add_argument("--job-id", required=True, help="Persona response job ID")
    parser.add_argument("--base-dir", default="runs", help="Base directory containing persona_responses/")
    parser.add_argument("--limit", type=int, default=200, help="Record limit for query")
    parser.add_argument("--no-records", action="store_true", help="Skip record payload in response")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    manager = PersonaResponseJobManager(base_dir)

    try:
        detail = manager.get_dashboard_run(args.job_id)
    except KeyError:
        raise SystemExit(f"[ERROR] Dashboard run '{args.job_id}' not found under {base_dir}")

    filters = DashboardFilters()
    response = manager.query_dashboard(
        args.job_id,
        filters=filters,
        limit=args.limit,
        include_records=not args.no_records,
    )

    summary = {
        "job_id": args.job_id,
        "project": detail.project_name,
        "cache_status": response.cache_status,
        "total_responses": response.total_responses,
        "filtered_responses": response.filtered_responses,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.no_records and response.records:
        print(f"\nSample records: {min(len(response.records), 3)} / {len(response.records)}")
        for row in response.records[:3]:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
