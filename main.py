from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langsmith import traceable
from typing import List, Optional
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Configure LangSmith BEFORE importing graph
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "resume-analyzer")

# Check if LangSmith is configured
LANGSMITH_CONFIGURED = bool(os.getenv("LANGCHAIN_API_KEY"))

if LANGSMITH_CONFIGURED:
    print("✅ LangSmith tracing ENABLED")
    print(f"   Project: {os.getenv('LANGCHAIN_PROJECT')}")
    print(f"   API Key: {os.getenv('LANGCHAIN_API_KEY')[:10]}...")
else:
    print("⚠️  LangSmith tracing DISABLED (LANGCHAIN_API_KEY not found)")
    print("   Set LANGCHAIN_API_KEY in .env to enable tracing")

from graph import (
    HFSpaceLLMClient,
    analyze_resume,
    AgentState
)


app = FastAPI(
    title="Resume Analyzer API",
    description="Upload resume and get skills extraction + job recommendations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResumeAnalysisResponse(BaseModel):
    """Response model for resume analysis"""
    skills: List[str]
    experience_level: str
    location: str
    job_results: List[str]
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    langsmith_enabled: bool


llm_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize LLM client on startup"""
    global llm_client
    
    BASE_URL = os.getenv("HF_SPACE_URL", "https://sameer-handsome173-coder-model-space.hf.space")
    ENDPOINT = os.getenv("HF_ENDPOINT", "/v1/generate")
    HF_TOKEN = os.getenv("HF_TOKEN", None)
    
    llm_client = HFSpaceLLMClient(
        base_url=BASE_URL,
        endpoint=ENDPOINT,
        token=HF_TOKEN
    )
    
    print("✓ LLM Client initialized")
    print(f"✓ Using endpoint: {BASE_URL}{ENDPOINT}")



@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API info"""
    return {
        "status": "active",
        "message": "Resume Analyzer API is running. Use /docs for API documentation.",
        "langsmith_enabled": LANGSMITH_CONFIGURED
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Service is operational",
        "langsmith_enabled": LANGSMITH_CONFIGURED
    }


@traceable(name="analyze_resume_endpoint", run_type="chain")
@app.post("/analyze-resume", response_model=ResumeAnalysisResponse)
async def analyze_resume_endpoint(
    file: UploadFile = File(...)
):
    """
    Upload resume (PDF, DOCX, or TXT) and get skills extraction + job recommendations
    
    - **file**: Resume file (PDF, DOCX, or TXT format)
    
    Returns:
    - Extracted skills
    - Experience level
    - Location preference
    - Relevant job postings
    """
    
    allowed_extensions = {".pdf", ".docx", ".txt"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        print(f"📄 Processing file: {file.filename}")
        print(f"📂 Temp path: {temp_file_path}")
        
        result = analyze_resume(temp_file_path, llm_client)
        
        response = ResumeAnalysisResponse(
            skills=result.get("extracted_skills", []),
            experience_level=result.get("experience_level", "mid"),
            location=result.get("location", "remote"),
            job_results=result.get("job_results", []),
            error=result.get("error", None)
        )
        
        print(f"✓ Analysis complete")
        print(f"✓ Skills found: {len(response.skills)}")
        print(f"✓ Jobs found: {len(response.job_results)}")
        
        return response
        
    except Exception as e:
        print(f"✗ Error processing resume: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing resume: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"Cleaned up temp file")
            except Exception as e:
                print(f"Warning: Could not delete temp file: {e}")



if __name__ == "__main__":
    import uvicorn
    
    if not os.getenv("RAPIDAPI_KEY"):
        os.environ["RAPIDAPI_KEY"] = "50949f7244mshbcb488b86e0c93bp12c41cjsn4bc33b509419"

    
    print("\n" + "="*60)
    print(" Starting Resume Analyzer API")
    print("="*60)
    print(f"📖 API Docs: http://localhost:8000/docs")
    print(f"🔍 LangSmith: {'ENABLED' if LANGSMITH_CONFIGURED else 'DISABLED'}")
    if LANGSMITH_CONFIGURED:
        print(f"📊 View traces at: https://smith.langchain.com/o/YOUR_ORG/projects/p/{os.getenv('LANGCHAIN_PROJECT')}")
    print("="*60 + "\n")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0",
        port=8000,
        reload=True
    )