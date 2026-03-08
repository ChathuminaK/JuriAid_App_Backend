from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.case_pdf import pdf_to_text
from app.case_law_pipeline import retrieve_case_law_from_case

router = APIRouter()


@router.post("/case-law/retrieve")
async def retrieve_case_law(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=20)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Upload a PDF file")

    pdf_bytes = await file.read()
    text = pdf_to_text(pdf_bytes)

    from app.api import case_law_engine
    return retrieve_case_law_from_case(case_law_engine, text, top_k=top_k)


@router.get("/case-law/{case_id}")
def get_case_law_detail(case_id: str):
    from app.api import case_law_engine

    if not case_law_engine.ready:
        raise HTTPException(status_code=500, detail="Case law engine not loaded")

    doc = case_law_engine.get_case_by_id(case_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Case law not found")

    return doc