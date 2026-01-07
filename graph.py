from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langsmith import traceable
import json
import re
import traceback
from typing import Dict
import os
import requests
import re
from typing import TypedDict, List, Annotated, Optional
from langgraph.graph import StateGraph, END
import PyPDF2
from docx import Document
import time


class AgentState(TypedDict):
    """State that gets passed between nodes"""
    resume_file_path: str
    resume_text: str
    extracted_skills: List[str]
    experience_level: str
    location: str
    job_results: List[str]
    messages: Annotated[list[BaseMessage], add_messages]
    error: str


class HFSpaceLLMClient:
    """Simple LLM client for Hugging Face Space endpoint"""
    
    def __init__(self, base_url: str, endpoint: str = "", token: Optional[str] = None):
        self.url = f"{base_url}{endpoint}" if endpoint else base_url
        self.token = token
        self.session = requests.Session()
        print(f"✓ Initialized LLM client for: {self.url}")
    
    @traceable(name="llm_generate")
    def generate(self, prompt: str, max_retries: int = 3, timeout: int = 120) -> str:
        """Generate text from LLM with retry logic"""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        payload = {"query": prompt}
        
        for attempt in range(max_retries):
            try:
                print(f"Calling LLM API (attempt {attempt + 1}/{max_retries})...")
                response = self.session.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=timeout
                )
                response.raise_for_status()
                
                result = response.json()
                answer = result.get("response", "")
                
                if not answer:
                    answer = result.get("generated_text", result.get("text", str(result)))
                
                print(f"✓ LLM response received (length: {len(answer)} chars)")
                return answer.strip()
                
            except requests.exceptions.Timeout:
                error_msg = f"LLM API timeout (attempt {attempt + 1}/{max_retries})"
                print(f"⚠️ {error_msg}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return f"ERROR: {error_msg}"
                    
            except requests.exceptions.HTTPError as e:
                error_msg = f"LLM API HTTP error: {e}"
                print(f"❌ {error_msg}")
                if attempt < max_retries - 1 and e.response.status_code >= 500:
                    wait_time = 2 ** attempt
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return f"ERROR: {error_msg}"
                    
            except Exception as e:
                error_msg = f"LLM generation failed: {e}"
                print(f"❌ {error_msg}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return f"ERROR: {error_msg}"
        
        return "ERROR: Max retries exceeded"
    
    def __del__(self):
        self.session.close()


@traceable(name="extract_pdf_text")
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"


@traceable(name="extract_docx_text")
def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error extracting DOCX: {str(e)}"


@traceable(name="extract_txt_text")
def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error extracting TXT: {str(e)}"


@traceable(name="job_search_api")
def job_search_tool(skills: List[str], location: str = "remote") -> List[str]:
    """Fetch job postings for multiple skills from JSearch API (RapidAPI)."""
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY"),
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    results: List[str] = []

    for skill in skills:
        querystring = {"query": f"{skill} in {location}", "num_pages": "1"}
        try:
            response = requests.get(url, headers=headers, params=querystring, timeout=30)
            if response.status_code == 200:
                jobs = response.json().get("data", [])
                if jobs:
                    top_jobs = [f"{skill}: {j['job_title']} at {j['employer_name']}" for j in jobs[:3]]
                    results.extend(top_jobs)
                else:
                    results.append(f"{skill}: No jobs found in {location}.")
            else:
                results.append(f"{skill}: API error - {response.status_code}")
        except Exception as e:
            results.append(f"{skill}: Request failed - {str(e)}")

    return results


@traceable(name="parse_json_from_llm")
def parse_json_strict(text: str) -> Dict:
    """Parse JSON with multiple fallback strategies"""
    # Check if response contains error
    if text.startswith("ERROR:"):
        raise ValueError(f"LLM returned error: {text}")
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    cleaned = text.strip()
    
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    cleaned = cleaned.strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end+1])
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"No valid JSON found in LLM output. Response: {text[:500]}")


@traceable(name="node_1_extract_profile")
def node_1_extract_profile(state: AgentState, llm_client: HFSpaceLLMClient) -> AgentState:
    """
    Node 1: Extract skills, experience level, and location from resume
    using ONE strict structured prompt (JSON only).
    """

    try:
        if not state.get("resume_text"):
            file_path = state["resume_file_path"]

            if file_path.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif file_path.endswith(".docx"):
                text = extract_text_from_docx(file_path)
            elif file_path.endswith(".txt"):
                text = extract_text_from_txt(file_path)
            else:
                state["error"] = "Unsupported file format"
                return state

            state["resume_text"] = text
            print(f"✓ Extracted resume text: {len(text)} characters")

        resume_text = state["resume_text"]
        
        # Truncate if too long (keep first 3000 chars to avoid timeout)
        if len(resume_text) > 3000:
            print(f"⚠️ Resume too long ({len(resume_text)} chars), truncating to 3000 chars")
            resume_text = resume_text[:3000]

        prompt = f"""
You are an information extraction system.

You MUST return ONLY valid JSON.
Do NOT include explanations, markdown, comments, or extra text.

====================
JSON SCHEMA (STRICT)
====================
{{
  "skills": [string],
  "experience_level": "entry | mid | senior | expert",
  "location": string
}}

====================
EXTRACTION RULES
====================

SKILLS:
- Extract ONLY technical skills
- Include: programming languages, frameworks, libraries, databases,
  tools, cloud platforms, DevOps tools, ML/AI tools, APIs, methodologies
- Do NOT include soft skills
- Remove duplicates
- Maximum 15 skills

EXPERIENCE_LEVEL:
Choose EXACTLY ONE:
- entry → 0–2 years, intern, junior
- mid → 3–5 years
- senior → 6–10 years
- expert → 10+ years

LOCATION:
- If city/country mentioned → return that
- If "remote", "WFH", or "work from home" → return "remote"
- If nothing mentioned → return "remote"

====================
RESUME TEXT
====================
{resume_text}

====================
FINAL OUTPUT
====================
Return ONLY valid JSON.
"""

        print("🔄 Extracting structured profile...")
        response = llm_client.generate(prompt, max_retries=3, timeout=120)
        print(f"📝 Raw LLM response (first 300 chars): {response[:300]}")

        data = parse_json_strict(response)

        skills = data.get("skills", [])
        skills = [s.strip() for s in skills if isinstance(s, str) and s.strip()]
        skills = list(dict.fromkeys(skills))[:15] 

        experience = data.get("experience_level", "mid").lower()
        if experience not in {"entry", "mid", "senior", "expert"}:
            experience = "mid"

        location = data.get("location", "remote").strip()
        if not location or len(location) > 50:
            location = "remote"

        state["extracted_skills"] = skills
        state["experience_level"] = experience
        state["location"] = location

        state["messages"].append(
            f"✓ Profile extracted | Skills: {len(skills)}, "
            f"Experience: {experience}, Location: {location}"
        )

        print(f"✅ Skills: {skills}")
        print(f"✅ Experience: {experience}")
        print(f"✅ Location: {location}")

    except Exception as e:
        print("❌ ERROR in node_1_extract_profile")
        print(traceback.format_exc())

        state["error"] = f"Profile extraction failed: {str(e)}"
        state["extracted_skills"] = []
        state["experience_level"] = "mid"
        state["location"] = "remote"

    return state


@traceable(name="node_2_find_jobs")
def node_2_find_jobs(state: AgentState) -> AgentState:
    """
    Node 2: Find relevant jobs using JSearch API
    """
    skills = state["extracted_skills"]
    location = state.get("location", "remote")
    
    if not skills:
        state["error"] = "No skills extracted, cannot search for jobs"
        state["job_results"] = []
        return state
    
    print(f"🔍 Searching jobs for skills: {', '.join(skills)}")
    
    job_results = job_search_tool(skills, location)
    state["job_results"] = job_results
    
    print(f"✅ Found {len(job_results)} job results")
    state["messages"].append(f"✓ Found {len(job_results)} job results")
    
    return state


def should_continue_to_jobs(state: AgentState) -> str:
    """Router: Check if we should continue to job search"""
    if state.get("error"):
        return "end"
    if not state.get("extracted_skills"):
        return "end"
    return "continue"


@traceable(name="build_resume_analyzer_graph")
def build_resume_analyzer_graph(llm_client: HFSpaceLLMClient):
    """Build the LangGraph workflow with 2 nodes"""
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("node_1_extract_skills", lambda state: node_1_extract_profile(state, llm_client))
    workflow.add_node("node_2_find_jobs", node_2_find_jobs)
    
    workflow.set_entry_point("node_1_extract_skills")
    
    workflow.add_conditional_edges(
        "node_1_extract_skills",
        should_continue_to_jobs,
        {
            "continue": "node_2_find_jobs",
            "end": END
        }
    )
    
    workflow.add_edge("node_2_find_jobs", END)
    
    return workflow.compile()


@traceable(name="analyze_resume")
def analyze_resume(resume_file_path: str, llm_client: HFSpaceLLMClient):
    """Main function to analyze resume and find jobs"""
    
    app = build_resume_analyzer_graph(llm_client)
    
    initial_state = {
        "resume_file_path": resume_file_path,
        "resume_text": "",
        "extracted_skills": [],
        "experience_level": "",
        "location": "remote",
        "job_results": [],
        "messages": [],
        "error": ""
    }
    
    result = app.invoke(initial_state)
    
    return result