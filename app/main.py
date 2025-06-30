from fastapi import FastAPI, UploadFile, File
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.config import CORS_ORIGINS
from app.services import process_pdfs, ask_question_service, get_processed_resumes, analyze_specific_resume

#create instance of fast api
app = FastAPI()

#cors to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload-and-process/")
async def upload_and_process(files: List[UploadFile] = File(...)):
    """Upload and process multiple PDF files for RAG"""
    print("Received files for processing:", files)
    num_chunks = await process_pdfs(files)
    return {"message": f"Processed PDFs into {num_chunks} chunks", "collection": "multi_pdf_rag"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question using RAG on the processed PDFs"""

    answer = ask_question_service(request.question)
    return {"answer": answer}

@app.get("/resumes")
async def get_resumes():
    """Get list of all processed resumes"""
    resumes = get_processed_resumes()
    return {"resumes": resumes}

@app.post("/analyze-resume")
async def analyze_resume(request: dict):
    """Analyze a specific resume against a job description"""
    resume_name = request.get('resume_name')
    job_description = request.get('job_description')
    
    if not resume_name or not job_description:
        return {"error": "Both resume_name and job_description are required"}
    
    analysis = analyze_specific_resume(resume_name, job_description)
    
    if analysis is None:
        return {"error": f"Resume '{resume_name}' not found"}
    
    return {"resume": resume_name, "analysis": analysis}