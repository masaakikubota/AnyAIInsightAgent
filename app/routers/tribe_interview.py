from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse

from ..dependencies import get_tribe_interview_job_manager
from ..models import (
    TribeInterviewJobConfig,
    TribeInterviewJobProgress,
    TribeInterviewJobResponse,
    TribeInterviewMode,
    TribeInterviewSheetNames,
)


router = APIRouter(prefix="/tribe-interview", tags=["tribe_interview"])


@router.get("", response_class=HTMLResponse)
def tribe_interview_index() -> str:
    app_dir = Path(__file__).resolve().parent.parent
    return (app_dir / "static" / "tribe_interview.html").read_text(encoding="utf-8")


@router.post("/jobs", response_model=TribeInterviewJobResponse)
async def create_tribe_interview_job(
    product_category: str = Form(...),
    country_region: str = Form(...),
    mode: str = Form("product"),
    persona_per_combination: int = Form(3),
    interviews_per_persona: int = Form(3),
    spreadsheet_url: str = Form(...),
    tribe_sheet_name: str = Form("Tribe_SetUp"),
    combination_sheet_name: str = Form("Tribe_Combination"),
    persona_sheet_name: str = Form("Persona_SetUp"),
    qa_llm_sheet_name: str = Form("QA_LLM"),
    qa_embedding_sheet_name: str = Form("QA_Embedding"),
    product_detail: str = Form(""),
    tagline_detail: str = Form(""),
    persona_prompt_template: str = Form(""),
    interview_questions: str = Form(""),
    images: UploadFile | List[UploadFile] | None = File(None),
    manager=Depends(get_tribe_interview_job_manager),
) -> TribeInterviewJobResponse:
    """Create a new tribe-interview job from form submission."""

    try:
        mode_enum = TribeInterviewMode(mode.lower())
    except ValueError as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid mode") from exc

    sheet_names = TribeInterviewSheetNames(
        tribe_setup=(tribe_sheet_name or "Tribe_SetUp").strip() or "Tribe_SetUp",
        tribe_combination=(combination_sheet_name or "Tribe_Combination").strip() or "Tribe_Combination",
        persona_setup=(persona_sheet_name or "Persona_SetUp").strip() or "Persona_SetUp",
        qa_llm=(qa_llm_sheet_name or "QA_LLM").strip() or "QA_LLM",
        qa_embedding=(qa_embedding_sheet_name or "QA_Embedding").strip() or "QA_Embedding",
    )

    questions = [line.strip() for line in interview_questions.splitlines() if line.strip()] if interview_questions else []

    config = TribeInterviewJobConfig(
        product_category=product_category.strip(),
        country_region=country_region.strip(),
        mode=mode_enum,
        persona_per_combination=max(1, min(10, int(persona_per_combination))),
        interviews_per_persona=max(1, min(10, int(interviews_per_persona))),
        sheet_names=sheet_names,
        spreadsheet_url=spreadsheet_url.strip(),
        product_detail=product_detail.strip() or None,
        tagline_detail=tagline_detail.strip() or None,
        persona_prompt_template=persona_prompt_template.strip() or None,
        interview_questions=questions or None,
    )

    response = await manager.create_job(config)

    upload_list: List[UploadFile] = []
    if isinstance(images, list):
        upload_list = [item for item in images if item and item.filename]
    elif isinstance(images, UploadFile) and images.filename:
        upload_list = [images]

    if upload_list:
        job_dir = manager.job_dir(response.job_id)
        image_dir = job_dir / "manual_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[str] = []
        for idx, upload in enumerate(upload_list, start=1):
            suffix = Path(upload.filename).suffix or ".png"
            dest = image_dir / f"manual_{idx:02d}{suffix}"
            with dest.open("wb") as fout:
                shutil.copyfileobj(upload.file, fout)
            upload.file.close()
            saved_paths.append(str(dest.relative_to(job_dir)))
        manager.update_config(response.job_id, image_paths=saved_paths)
        await manager.add_artifacts(response.job_id, {"images": ", ".join(saved_paths)})

    return response


@router.get("/jobs/{job_id}", response_model=TribeInterviewJobProgress)
async def get_tribe_interview_job(
    job_id: str,
    manager=Depends(get_tribe_interview_job_manager),
) -> TribeInterviewJobProgress:
    """Placeholder endpoint for retrieving tribe-interview job progress."""

    try:
        return manager.get_progress(job_id)
    except KeyError as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found") from exc


__all__ = ["router"]
