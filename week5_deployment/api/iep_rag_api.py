"""
IEP Goal Generator RAG API
FastAPI application for generating IEP goals based on student info.

Usage:
    python week5_deployment/api/iep_rag_api.py
    # or
    uvicorn week5_deployment.api.iep_rag_api:app --reload --port 8000
"""

import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define persistent storage path
VECTORSTORE_PATH = str(Path(__file__).parent.parent / "vectorstore")

load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="IEP Goal Generator API",
    description="Generate IEP goals using a RAG system",
    version="1.0.0"
)

# --- Global Variables ---
# These will be loaded at startup
llm_client = None
embedding_model = None
ooh_collection = None
standards_collection = None
sample_goals_collection = None
# --- End Global Variables ---


@app.on_event("startup")
async def load_models():
    """Load models and vector store on startup."""
    global llm_client, embedding_model, ooh_collection, standards_collection, sample_goals_collection
    
    logger.info("Loading models and vector store...")
    
    # 1. Load LLM Client (from .env)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise ValueError("OPENAI_API_KEY not set")
    llm_client = OpenAI(api_key=api_key)

    # 2. Load Embedding Model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Load Persistent Vector Store
    if not Path(VECTORSTORE_PATH).exists():
        logger.error(f"Vector store not found at {VECTORSTORE_PATH}")
        logger.error("Please run `python week5_deployment/data/build_vectorstore.py` first.")
        raise FileNotFoundError(f"Vector store not found at {VECTORSTORE_PATH}")
        
    client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    
    ooh_collection = client.get_collection(name="ooh_collection")
    standards_collection = client.get_collection(name="standards_collection")
    sample_goals_collection = client.get_collection(name="sample_goals_collection")
    
    logger.info("Models and vector store loaded successfully!")


# --- Pydantic Request/Response Models ---
class StudentInfo(BaseModel):
    profile_text: str
    career_interest: str

class GoalOutput(BaseModel):
    generated_goals: str
    retrieved_ooh_context: str
    retrieved_standards_context: str

# --- Helper Functions ---
def retrieve_context(query: str, collection, top_k=3) -> str:
    """Retrieves context from a specific ChromaDB collection."""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return "\n---\n".join(results['documents'][0])
    except Exception as e:
        logger.error(f"Error retrieving from collection {collection.name}: {e}")
        return "Error: Could not retrieve context."

def build_final_prompt(student_profile, ooh_context, standards_context, sample_goals) -> str:
    """Builds the final, complex prompt for the LLM."""
    
    return f"""
    You are an expert in Special Education and transition planning. Your task is to create a set of measurable, high-quality IEP goals based on the provided student profile and context.

    The goals *must* be grounded in and aligned with the provided Occupational Outlook Handbook (OOH) context and State Educational Standards.

    --- STUDENT PROFILE ---
    {student_profile}

    --- OOH CAREER CONTEXT ---
    Here is information on the student's career interests from the Occupational Outlook Handbook:
    {ooh_context}

    --- STATE EDUCATIONAL STANDARDS ---
    Here are relevant state standards for workplace skills:
    {standards_context}

    --- SAMPLE GOALS (for structure) ---
    {sample_goals}

    --- TASK ---
    Based *only* on the profile and the context provided, generate the following for this student. Use the "Case Study: Clarence" output format as a guide.

    1.  **Measurable Postsecondary Goal (Employment):**
        (Must be a specific, measurable goal related to the student's interest and OOH context)

    2.  **Measurable Postsecondary Goal (Education/Training):**
        (Must be a specific, measurable goal related to on-the-job training or education needed, based on OOH context)

    3.  **Measurable Annual Goal:**
        (This goal *must* be a stepping stone to the postsecondary goals and *must* align with the State Standards context. Make it measurable, e.g., "In 36 weeks... in 4 out of 5 opportunities.")

    4.  **Alignment to Standards:**
        (Explicitly state which OOH standards (e.g., "Retail Sales Worker skills") and State Standards are met by this annual goal. Quote or paraphrase the standard.)

    5.  **Short-Term Objectives/Benchmarks (3):**
        (List 3 small, sequential steps that build up to the annual goal.)

    Provide the output in clean Markdown.
    """

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "IEP Goal Generator API",
        "status": "running",
        "endpoints": ["/generate-goals", "/health"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = (
        llm_client is not None and
        embedding_model is not None and
        ooh_collection is not None
    )
    return {
        "status": "healthy" if model_loaded else "degraded",
        "models_loaded": model_loaded
    }

@app.post("/generate-goals", response_model=GoalOutput)
async def generate_goals(student_info: StudentInfo):
    """Generate IEP goals based on student profile and career interest."""
    try:
        # 1. "Router" RAG: Find career info
        logger.info(f"Retrieving OOH info for: {student_info.career_interest}")
        ooh_context = retrieve_context(
            query=f"Information about {student_info.career_interest} occupation",
            collection=ooh_collection
        )
        
        # 2. "Skills" RAG: Find standards
        logger.info("Retrieving standards info...")
        standards_context = retrieve_context(
            query=f"Educational standards for skills needed in {student_info.career_interest}: {ooh_context}",
            collection=standards_collection
        )
        
        # 3. Retrieve sample goals
        logger.info("Retrieving sample goals...")
        sample_goals = retrieve_context(
            query="Examples of postsecondary and annual goals",
            collection=sample_goals_collection
        )

        # 4. Final Generation
        logger.info("Generating final goals...")
        final_prompt = build_final_prompt(
            student_info.profile_text,
            ooh_context,
            standards_context,
            sample_goals
        )
        
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo", # Use a fast and cheap model
            messages=[
                {"role": "system", "content": "You are an expert Special Education professional specializing in transition planning and IEP goal development. You must follow IDEA 2004 requirements."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.5,
        )
        
        generated_text = response.choices[0].message.content

        return GoalOutput(
            generated_goals=generated_text,
            retrieved_ooh_context=ooh_context,
            retrieved_standards_context=standards_context
        )
        
    except Exception as e:
        logger.error(f"Error generating goals: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)