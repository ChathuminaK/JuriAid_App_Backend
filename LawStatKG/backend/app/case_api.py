from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from typing import Optional
from app.case_pdf import pdf_to_text
from app.case_law_pipeline import retrieve_case_applicable_laws

router = APIRouter()

@router.post("/case/laws")
async def case_laws(
    file: UploadFile = File(...),
    as_of_date: Optional[str] = Query("today"),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Upload a PDF file")

    pdf_bytes = await file.read()
    text = pdf_to_text(pdf_bytes)

    from app.api import engine, kg

    return retrieve_case_applicable_laws(
        engine=engine,
        kg_client=kg,
        case_text=text,
        as_of_date=as_of_date
    )