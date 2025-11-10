"""
IEP Goal Generator - Streamlit UI

A simple Streamlit interface that interacts with the
backend IEP RAG API.

Usage:
    streamlit run week5_deployment/ui/iep_generator_app.py
"""

import streamlit as st
import requests
import json
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint
#API_URL = "http://localhost:8000/generate-goals" # Assumes API is on port 8000
API_URL = os.getenv("API_URL", "http://localhost:8000/generate-goals")

# --- Page Configuration ---
st.set_page_config(
    page_title="IEP Goal Generator",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 RAG-Based IEP Goal Generator")
st.markdown("""
    This tool helps special education professionals create measurable, standards-aligned
    IEP goals for students.
    
    1.  Enter the student's profile information.
    2.  Add their primary career interest.
    3.  Click "Generate" to get AI-assisted, standards-aligned goals.
""")

# --- User Input Form ---
st.header("1. Student Information")

case_study_clarence = """Student Information:
- 15-year-old sophomore with a behavior disorder
- Completed the O*Net Interest Profiler assessment
- Shows strong interest in the "Enterprising" category
- Prefers hands-on learning over academic instruction

Assessment Results:
- O*Net Interest Profiler indicates strength in Enterprising activities
- Career suggestions include retail salesperson or driver/sales worker
- Student interview ("Vision for the Future") indicates interest in working at Walmart
"""

profile_text = st.text_area(
    "Student Profile (assessment results, interests, needs):",
    value=case_study_clarence,
    height=250
)

career_interest = st.text_input(
    "Primary Career Interest (e.g., 'Retail Sales', 'Driver/Sales Worker'):",
    value="Retail Sales Worker"
)

# --- Generate Button & API Call ---
if st.button("Generate IEP Goals", type="primary"):
    if not profile_text or not career_interest:
        st.error("Please fill out both the student profile and career interest.")
    else:
        with st.spinner("Generating goals... This may take a moment."):
            try:
                # Prepare payload for the API
                payload = {
                    "profile_text": profile_text,
                    "career_interest": career_interest
                }
                
                # Make POST request to the FastAPI backend
                response = requests.post(API_URL, json=payload, timeout=120)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.header("✅ 2. Generated IEP Goals & Objectives")
                    st.markdown(data['generated_goals'])
                    
                    st.header("🔍 3. Alignment & Explainability")
                    st.markdown("The generated goals are grounded in the following retrieved context:")
                    
                    with st.expander("Show Retrieved Occupational Outlook Handbook (OOH) Context"):
                        st.text(data['retrieved_ooh_context'])
                        
                    with st.expander("Show Retrieved State Educational Standards Context"):
                        st.text(data['retrieved_standards_context'])
                        
                else:
                    st.error(f"Error from API (Status {response.status_code}):")
                    st.json(response.json())

            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Could not connect to the API.")
                st.error(f"Please ensure the API is running at: {API_URL}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")