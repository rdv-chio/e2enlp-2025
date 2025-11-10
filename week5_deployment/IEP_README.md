# Week 5: End-to-End NLP Deployment

This week covers deploying production-ready NLP applications with APIs, UIs, and Docker containers. This directory contains the **Final Project: IEP Goal Generator** as well as other course demos.

## 🚀 Final Project: IEP Goal Generator

This is the main project for the course. It is a full end-to-end RAG application, containerized with Docker.

### 1. Build the Vector Store (One-Time Setup)

First, you must process your knowledge base (OOH, standards) into a vector database.

1.  **Add Data:**
    * Place your saved `ooh_retail.html` and `ooh_driver.html` files into `week5_deployment/data/source_docs/`.
    * Place your `standards.pdf` file into `week5_deployment/data/source_docs/`.
2.  **Build the Database:**
    ```bash
    # (Activate your conda env first)
    # This script will scrape/parse local files and create the vectorstore/ directory
    python3 week5_deployment/data/build_vectorstore.py
    ```

### 2. Run with Docker (Recommended)

This is the simplest way to run the full application.

1.  **Ensure `.env` is ready:** Make sure your `OPENAI_API_KEY` is in the `.env` file in the project root.
2.  **Run Docker Compose:**
    ```bash
    # From the project root (e2enlp-2025/)
    docker-compose -f week5_deployment/docker/docker-compose.iep.yml --env-file ./.env up --build
    ```
3.  **Access the App:**
    * **UI:** [http://localhost:8501](http://localhost:8501)
    * **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Run Locally (For Debugging)

**Terminal 1: Run the API**
```bash
# (Activate your conda env)
# (Ensure .env file is in the root)
python3 week5_deployment/api/iep_rag_api.py