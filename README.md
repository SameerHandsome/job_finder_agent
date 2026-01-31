README — Resume Analyzer API
============================

What I did and how I produced this README
----------------------------------------
I scanned the repository source files (main.py and graph.py) and extracted the runtime behavior, API surface, graph workflow, LLM client implementation, and environment configuration that the code contains. Below is a README generated strictly from the code (no external assumptions).

Project overview
----------------
Resume Analyzer API accepts a resume file (PDF/DOCX/TXT), extracts technical skills, infers experience level and location, and returns those fields along with a list of relevant job results. The project uses a small LLM client that calls a Hugging Face Space endpoint and a LangGraph-style workflow (built and executed in graph.py) to coordinate extraction and job-finding steps. The service is exposed via a FastAPI application.

Key features (from code)
- HTTP API with endpoints: / (info), /health, /analyze-resume
- Upload resume files (.pdf, .docx, .txt)
- Extract resume text (PyPDF2 for PDF, python-docx for .docx, plain read for .txt)
- Extract structured profile (technical skills, experience level, location) using an LLM prompt that requires JSON-only output
- A LangGraph workflow coordinates nodes to produce job_results and messages
- LLM client targets a Hugging Face Space endpoint with retry and backoff logic
- Optional LangSmith tracing via langsmith.traceable integration

Architecture and logic flow (strictly from the code)
----------------------------------------------------

High-level flow
1. FastAPI app (main.py) runs and initializes an HFSpaceLLMClient on startup.
2. Client uploads a resume to POST /analyze-resume as multipart/form-data; server writes the file to a temporary file.
3. analyze_resume (graph.py) builds a LangGraph (2-node workflow, per comments) and invokes it with an initial state that contains:
   - resume_file_path, resume_text, extracted_skills, experience_level, location, job_results, messages, error
4. Node 1 (node_1_extract_profile) — Extract profile:
   - Ensures resume text is present by reading the temporary file and using:
     - extract_text_from_pdf (uses PyPDF2)
     - extract_text_from_docx (uses docx.Document)
     - extract_text_from_txt (simple read)
   - Truncates resume_text to first 3000 characters if longer to avoid LLM timeouts
   - Builds a strict JSON-only prompt describing a JSON schema:
     { "skills": [string], "experience_level": "entry | mid | senior | expert", "location": string }
     - Extraction rules: only technical skills, max 15, no soft skills; experience level mapping and location rules are spelled out in the prompt
   - Calls the HFSpaceLLMClient.generate(prompt, ...) and parses the LLM response with parse_json_strict (a parser helper in graph.py)
   - Deduplicates and sanitizes skills, normalizes experience to one of entry/mid/senior/expert, validates/limits location (defaulting to "remote")
   - Writes extracted_skills, experience_level, location back into state and appends a status message to state["messages"]
5. Node 2 (not fully shown in the excerpts) — Based on repository state and keys present:
   - The workflow has a second node (code comments say "2 nodes") which ultimately populates state["job_results"] and returns the final state. The analyze_resume function returns the invoked result to main.py.
6. main.analyze_resume_endpoint:
   - Receives the result from analyze_resume, maps fields into ResumeAnalysisResponse Pydantic model:
     - skills: List[str]
     - experience_level: str
     - location: str
     - job_results: List[str]
     - error: Optional[str]
   - Cleans up temporary file in a finally block.

LLM client (HFSpaceLLMClient)
- Constructed with base_url (default in code: "https://sameer-handsome173-coder-model-space.hf.space") and endpoint (default "/v1/generate") and token optional.
- Uses requests.Session
- generate(prompt, max_retries=3, timeout=120) implements retry/backoff:
  - Handles HTTP errors (status codes) and general exceptions
  - Differentiates retriable errors (server errors) and non-retriable
  - Logs error messages and returns "ERROR: ..." strings if final failure

Text extraction helpers (graph.py)
- extract_text_from_pdf(file_path) — uses PyPDF2
- extract_text_from_docx(file_path) — uses docx.Document
- extract_text_from_txt(file_path) — opens file utf-8 and reads content

Tracing / Observability
- The code uses langsmith.traceable decorators in multiple functions, and environment variable LANGCHAIN_API_KEY controls LangSmith tracing enabling. The code sets the following LangChain/LangSmith env defaults:
  - LANGCHAIN_TRACING_V2 (defaults to "true" if not set)
  - LANGCHAIN_ENDPOINT (defaults to "https://api.smith.langchain.com")
  - LANGCHAIN_PROJECT (defaults to "resume-analyzer")
- When LANGCHAIN_API_KEY is present the application prints LangSmith tracing is enabled and shows project and API key prefix.

Tech stack and dependencies (inferred strictly from imports)
- Python (3.x)
- FastAPI — HTTP server and API models
- Uvicorn — ASGI server (used in main.py)
- pydantic — request/response models
- dotenv (python-dotenv) — load environment variables
- requests — HTTP client (used in HFSpaceLLMClient)
- PyPDF2 — PDF text extraction
- python-docx (docx) — .docx extraction
- langsmith — traceable decorater and tracing integration
- langgraph (langgraph.graph, message) — state graph / workflow primitives (StateGraph, END)
- langchain_core.messages.BaseMessage — message types used by the graph
- typing, pathlib, tempfile, os, json, time, traceback, re — stdlib usage

Files of interest (from repository)
- main.py — FastAPI app, startup logic, API endpoints, temp file handling, environment defaults
- graph.py — LLM client, text extraction, graph nodes (node_1_extract_profile), analyze_resume orchestration

Configuration and environment variables (as seen in code)
- .env is loaded at startup (load_dotenv())
- LANGCHAIN_TRACING_V2 (default "true" if not set)
- LANGCHAIN_ENDPOINT default: "https://api.smith.langchain.com"
- LANGCHAIN_PROJECT default: "resume-analyzer"
- LANGCHAIN_API_KEY — if set, enables LangSmith tracing
- HF_SPACE_URL — base URL for HF Space LLM client; default in code:
  https://sameer-handsome173-coder-model-space.hf.space
- HF_ENDPOINT — endpoint path for HF Space; default "/v1/generate"
- HF_TOKEN — optional token passed to HFSpaceLLMClient
- RAPIDAPI_KEY — if not set, main.py explicitly sets a default value in __main__:
  50949f7244mshbcb488b86e0c93bp12c41cjsn4bc33b509419
  (the code sets this value at process start if not present)

Security note (from code)
- The repository code sets a fallback RAPIDAPI_KEY in main.py when run as __main__ if no RAPIDAPI_KEY is present in the environment. If deploying publicly, remove or replace hard-coded credentials and set secrets via environment variables.

API overview (based directly on code)
- GET /          -> returns service info (status, message, langsmith_enabled)
- GET /health    -> returns health check JSON (status, message, langsmith_enabled)
- POST /analyze-resume
  - Accepts a form file field named file (UploadFile)
  - Allowed file extensions: .pdf, .docx, .txt
  - Returns (ResumeAnalysisResponse):
    - skills: List[str]
    - experience_level: str
    - location: str
    - job_results: List[str]
    - error: Optional[str]

Example request (from code behavior)
- Using curl to upload a resume file:
  curl -F "file=@/path/to/resume.pdf" http://localhost:8000/analyze-resume
- Response: JSON with fields shown above (skills, experience_level, location, job_results, error)

Behavioral/operational details noted in code
- Resume text longer than 3000 characters is truncated before calling the model
- LLM responses are expected to be valid JSON for node_1 prompts; parse_json_strict is used to enforce that
- Temporary files are cleaned up in a finally block after processing
- Errors result in HTTP 500 from the endpoint with a detail string indicating the error
- Detailed logs are printed to stdout for lifecycle events, node outputs, and errors

How to run locally (instructions based on code)
1. Install dependencies (minimum inferred)
   pip install fastapi uvicorn python-dotenv requests PyPDF2 python-docx langsmith langgraph langchain-core pydantic

   Note: The repository imports packages named "langgraph", "langsmith", and "langchain_core". Ensure matching package names/versions for your environment.

2. Provide environment variables (optional)
   - Create a .env file or export env vars:
     - LANGCHAIN_API_KEY (to enable LangSmith tracing)
     - HF_SPACE_URL (if you want a different HF Space host; otherwise the code uses a default)
     - HF_ENDPOINT (optional; default /v1/generate)
     - HF_TOKEN (optional)
     - RAPIDAPI_KEY (optional; code sets a fallback default when running main directly)
   - Example .env:
     LANGCHAIN_API_KEY=your_langchain_api_key
     HF_SPACE_URL=https://your-hf-space.hf.space
     HF_ENDPOINT=/v1/generate
     HF_TOKEN=your_token
     RAPIDAPI_KEY=your_rapidapi_key

3. Start the app
   - Option A: Run via python (the repository includes an if __name__ == "__main__": uvicorn.run(...) block)
     python main.py
     The code prints:
       Starting Resume Analyzer API
       API Docs: http://localhost:8000/docs
     (The code uses port 8000 for docs URL)

   - Option B: Run with uvicorn directly (recommended for development)
     uvicorn main:app --reload --host 0.0.0.0 --port 8000

4. Use the API
   - Open http://localhost:8000/docs for interactive Swagger UI (FastAPI auto-docs)
   - POST a resume to /analyze-resume to get the JSON response

Limitations and notes (derived from code)
- Only PDF, DOCX, TXT resume formats are accepted (explicit file extension check)
- The extraction node enforces strict JSON output from the LLM. If the model returns non-JSON text the code attempts to parse with parse_json_strict; parse failures will set an error in state and propagate to the endpoint.
- The code truncates resume text to 3000 characters before extraction to avoid LLM timeouts (explicit in node_1 logic)
- The LLM client wraps Hugging Face Space-style endpoints — the actual model behavior depends on the Space you point the client to
- The second graph node produces job_results but the exact data source (LLM or external API) is not fully shown in the scanned excerpts; the final response includes job_results as a list of strings


Contact / Attribution
----------------------
This README was generated by scanning the repository code (main.py and graph.py) and describing behavior strictly as implemented in the code. If you want the README saved into a README.md file in the repository or want help producing a Dockerfile or requirements file, tell me and I will create them next.
