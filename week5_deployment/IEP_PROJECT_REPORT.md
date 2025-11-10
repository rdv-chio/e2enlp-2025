# Final Project Report: RAG-Based IEP Goal Generator

This report documents the design, implementation, and evaluation of the RAG-based IEP Goal Generator, fulfilling the final project requirements for the End-to-End NLP course.

## 1. Project Requirements

The system successfully meets all project requirements:
- **CLO 2 (RAG & Prompting):** Implemented a multi-step RAG pipeline with sophisticated prompt engineering.
- **CLO 4 (End-to-End Pipeline):** Designed a full pipeline including data collection (`build_vectorstore.py`), modeling (`iep_rag_api.py`), evaluation (this report), and deployment (`docker-compose.iep.yml`, `iep_generator_app.py`).

## 2. Data Collection and Preprocessing (Requirement #1)

- **Data Sources:**
    - **Occupational Outlook Handbook (OOH):** To overcome anti-scraping protections (403 Forbidden errors), the target HTML pages (`retail-sales-workers.htm`, `driver-sales-workers.htm`) were saved locally.
    - **State Standards:** The "P21 Framework Definitions" PDF was used as the source for educational standards.
    - **Sample IEP Goals:** A small set of 2 high-quality goals was hard-coded in `scrapers.py` to be used as few-shot examples for the generator.

- **Text Extraction & Chunking:**
    - A custom script (`scrapers.py`) was created to handle data loading.
    - `BeautifulSoup` was used to parse the local OOH HTML files, extracting the "What They Do" and "Important Qualities" sections.
    - `PyPDF` was used to extract text from all pages of the standards PDF.
    - `RecursiveCharacterTextSplitter` was used with a `chunk_size` of 500 and `overlap` of 50. This provided chunks small enough for the context window but large enough to retain meaning.

- **Embeddings & Storage:**
    - **Model:** `all-MiniLM-L6-v2` (from `sentence-transformers`) was used as a fast, high-quality, and local embedding model.
    - **Database:** `ChromaDB` was used as a persistent, on-disk vector store.
    - **Strategy:** Three separate collections were created (`ooh_collection`, `standards_collection`, `sample_goals_collection`) to allow for targeted retrieval.

## 3. RAG Pipeline and Prompt Engineering (Requirements #2 & #3)

Our RAG pipeline is a **multi-step retrieval chain** implemented in `iep_rag_api.py`:

1.  **"Career" Retrieval:** The user's `career_interest` (e.g., "Retail Sales Worker") is used to query *only* the `ooh_collection`.
2.  **"Skills" Retrieval:** The retrieved OOH context is *then* used to query the `standards_collection`. This finds educational standards that are semantically relevant to the job's skills (e.g., "customer service" in OOH matches "communication" in standards).
3.  **"Example" Retrieval:** A generic query retrieves few-shot examples from the `sample_goals_collection` to show the model the desired output format.

This multi-step approach is our core "retrieval strategy" and is highly effective.

### Prompt Engineering

The final prompt (in `build_final_prompt`) is a zero-shot, role-based prompt that is dynamically "stuffed" with our retrieved context.
- **Role:** `You are an expert in Special Education and transition planning...`
- **Context:** The prompt clearly separates the `STUDENT PROFILE`, `OOH CONTEXT`, `STATE STANDARDS`, and `SAMPLE GOALS`.
- **Task:** The model is given a strict 5-part structure to follow, ensuring all required outputs (postsecondary goals, annual goal, alignment, benchmarks) are generated.

## 4. User Interface & Deployment (Requirement #4 & CLO 4)

- **User Interface:** A `Streamlit` application (`ui/iep_generator_app.py`) provides the UI. It correctly accepts all required student information and displays the final goals. It provides **explainability** by showing the retrieved OOH and Standards context in expander sections, fulfilling the "Shows the alignment" requirement.
- **Deployment:** The entire application is containerized using `docker-compose`.
    - `docker-compose.iep.yml` defines two services: `api` and `ui`.
    - `Dockerfile.rag` and `Dockerfile.ui` containerize the FastAPI backend and Streamlit frontend, respectively.
    - The API's `healthcheck` was configured with a `start_period` to allow the models time to load before Docker marked the container as healthy.
    - The UI's `API_URL` is set via an environment variable in the compose file, allowing it to find the API container using Docker's internal DNS (`http://api:8000`).

## 5. Evaluation and Analysis (Requirement #5)

### Evaluation of System Performance

The system was tested using the "Case Study: Clarence" sample data.

- **Input:** Clarence, 15, "Enterprising" interest, "Retail Sales Worker".
- **Retrieval (from logs):**
    - `api-1  | INFO:iep_rag_api:Retrieving OOH info for: Retail Sales Worker` (Success)
    - `api-1  | INFO:iep_rag_api:Retrieving standards info...` (Success)
- **Generation (from UI):**
    - The system successfully generated goals highly similar to the "Expected Output" from the project description.
    - **Postsecondary Goal:** `After high school, Clarence will obtain part-time employment at a retail store, such as Walmart, as a sales associate.` (PASS)
    - **Annual Goal:** `In 36 weeks, Clarence will demonstrate effective workplace communication... by appropriately greeting customers, listening, and responding to questions in 4 out of 5 observed opportunities.` (PASS)
    - **Alignment:** The model correctly stated alignment with "OOH standards for Retail Sales Workers" and "21st Century Skills: Communication." (PASS)

### Strengths and Limitations

- **Strengths:**
    - **Highly Aligned:** The multi-step RAG chain ensures goals are not just generic, but *directly* grounded in both industry and state standards.
    - **Explainable:** The UI shows the user *why* a goal was generated.
    - **Production-Ready & Scalable:** The system is fully containerized with Docker, separating the API and UI. This is a robust, production-grade pattern.
    - **Offline Data:** By parsing local HTML files, the data pipeline is resilient to network errors or website anti-scraping measures.

- **Limitations:**
    - **Limited Knowledge Base:** The vector store only contains data for the careers we manually save. A user asking for "plumber" would get poor results.
    - **Static Data:** The OOH data is only as fresh as the last `build_vectorstore.py` run.

### Potential Improvements

- **Expand Data Pipeline:** The `OOH_FILES` list in `build_vectorstore.py` can be expanded to include 50+ saved HTML files to cover more careers.
- **Quantitative Evaluation:** A formal test set of 20-30 student profiles could be created. We could write "golden" human-generated goals for each and use the `week4_evaluation/sequence_metrics.py` script to get ROUGE scores for our model's outputs, giving us a quantitative quality metric.
- **Integrate Utils:** The `week5_deployment/utils` helpers can be fully integrated into `iep_rag_api.py` to add structured logging, request timing, and error handling.