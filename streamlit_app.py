import streamlit as st

# MUST BE FIRST
st.set_page_config(page_title="Pathfinder AI", page_icon="🧭")

import google.generativeai as genai
import os
import requests
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader

# Load API keys
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
JSEARCH_API_KEY = st.secrets.get("JSEARCH_API_KEY", "")

genai.configure(api_key=GOOGLE_API_KEY)

# Load Gemini model
model = genai.GenerativeModel("models/gemini-1.5-pro-002")

# PDF Resume Upload
uploaded_text = ""
uploaded_file = st.file_uploader("Upload Resume (PDF optional)", type="pdf")
if uploaded_file:
    reader = PdfReader(uploaded_file)
    uploaded_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Dynamically create vectorstore
@st.cache_resource
def create_vectorstore():
    fallback_text = "Experienced professional with a background in strategy, marketing, and AI-driven content development."
    docs = [Document(page_content=uploaded_text or fallback_text)]

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    return FAISS.from_documents(texts, embeddings)

vectorstore = create_vectorstore()
retriever = vectorstore.as_retriever()

# UI and Form
st.title("🧭 Pathfinder AI: Career Coach")
st.markdown("Get resume summaries + job fit insights powered by Gemini.")

with st.form("input_form"):
    name = st.text_input("Name")
    role = st.text_input("Current Role")
    skills = st.text_input("Key Skills")
    goal = st.text_input("Career Goal")
    tone = st.selectbox(
        "Choose a tone for your resume summary",
        ["professional", "confident", "enthusiastic", "strategic", "creative"]
    )
    location = st.text_input("Job Search Location (optional)")
    submit = st.form_submit_button("Generate")

if submit:
    with st.spinner("Generating resume and retrieving job insights..."):
        prompt = f"""
        Write a {tone} professional resume summary for {name}, currently a {role}, \
        with skills in {skills}, seeking a role in {goal}.
        """
        response = model.generate_content(prompt)
        summary = response.text.strip()

        st.subheader("Generated Resume Summary")
        st.success(summary)

        if JSEARCH_API_KEY.strip() and goal:
            st.subheader("🔎 Real-Time Job Listings")
            job_api_url = "https://jsearch.p.rapidapi.com/search"
            headers = {
                "X-RapidAPI-Key": JSEARCH_API_KEY,
                "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
            }
            query = f"{goal} in {location}" if location else goal
            params = {"query": query, "page": "1", "num_pages": "1"}

            try:
                response = requests.get(job_api_url, headers=headers, params=params)
                response.raise_for_status()
                results = response.json().get("data", [])
                if results:
                    for job in results[:5]:
                        st.markdown(f"**{job['job_title']}** at *{job['employer_name']}*")
                        st.caption(f"{job['job_city']}, {job['job_state']} | {job['job_employment_type']}")
                        st.write(job['job_description'][:250] + "...")
                        st.markdown(f"[View Job Posting]({job['job_apply_link']})")
                        st.markdown("---")
                else:
                    st.info("No job matches found for your role yet — try another search.")
            except Exception as e:
                st.error(f"Job search failed: {str(e)}")
        else:
            st.caption("⚠️ Job search API not configured or query missing. Add your JSEARCH_API_KEY to enable this feature.")

st.caption("Created by Alison Morano | Powered by Gemini 1.5 + FAISS + LangChain + JSearch")
