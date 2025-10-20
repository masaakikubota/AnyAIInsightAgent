from pathlib import Path

from app.models import JobStatus, RunConfig
from app.worker import Job, JobManager


def _make_job(tmp_path: Path, *, mode: str, enable_ssr: bool) -> tuple[JobManager, Job]:
    cfg = RunConfig(
        spreadsheet_url="https://docs.google.com/spreadsheets/d/example",
        mode=mode,
        enable_ssr=enable_ssr,
    )
    job = Job(job_id="job-1", cfg=cfg, status=JobStatus.pending)
    manager = JobManager(tmp_path)
    return manager, job


def test_csv_mode_ssr_uses_100_row_chunks(tmp_path):
    manager, job = _make_job(tmp_path, mode="csv", enable_ssr=True)

    manager._adjust_mode_defaults(job)

    assert job.cfg.batch_size == 10  # unchanged
    assert job.cfg.max_category_cols == 200
    assert job.cfg.sheet_chunk_rows == 100
    assert job.cfg.chunk_row_limit == 100
    assert job.cfg.writer_flush_batch_size == 100


def test_csv_mode_non_ssr_uses_500_row_chunks(tmp_path):
    manager, job = _make_job(tmp_path, mode="csv", enable_ssr=False)

    manager._adjust_mode_defaults(job)

    assert job.cfg.batch_size == 10
    assert job.cfg.sheet_chunk_rows == 500
    assert job.cfg.chunk_row_limit == 500
    assert job.cfg.writer_flush_batch_size == 500


def test_video_mode_ssr_forces_single_definition(tmp_path):
    manager, job = _make_job(tmp_path, mode="video", enable_ssr=True)

    manager._adjust_mode_defaults(job)

    assert job.cfg.batch_size == 1
    assert job.cfg.max_category_cols == 1
    assert job.cfg.sheet_chunk_rows == 50
    assert job.cfg.chunk_row_limit == 50
    assert job.cfg.writer_flush_batch_size == 50


def test_video_mode_non_ssr_forces_single_definition(tmp_path):
    manager, job = _make_job(tmp_path, mode="video", enable_ssr=False)

    manager._adjust_mode_defaults(job)

    assert job.cfg.batch_size == 1
    assert job.cfg.max_category_cols == 1
    assert job.cfg.sheet_chunk_rows == 100
    assert job.cfg.chunk_row_limit == 100
    assert job.cfg.writer_flush_batch_size == 100
