import logging
from pathlib import Path

import pytest

from app.worker import Job, JobManager
from app.models import RunConfig


@pytest.mark.asyncio
async def test_ssr_concurrency_guard_forces_batch_one(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    manager = JobManager(tmp_path)
    cfg = RunConfig(
        spreadsheet_url="https://docs.google.com/spreadsheets/d/test",
        sheet_keyword="Sheet",
        score_sheet_keyword="Score",
        spreadsheet_id="spreadsheet-id",
        sheet_name="Sheet1",
        score_sheet_name="Score1",
        enable_ssr=True,
        batch_size=8,
    )
    job = Job(job_id="job123", cfg=cfg)

    caplog.set_level(logging.DEBUG)
    manager._adjust_mode_defaults(job)

    assert job.cfg.batch_size == 1
    assert any("evt=ssr_concurrency_forced" in record.message for record in caplog.records)
