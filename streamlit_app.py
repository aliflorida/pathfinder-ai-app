import streamlit as st
import google.generativeai as genai
import os
import requests
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader

# Load API key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
JSEARCH_API_KEY = st.secrets.get("JSEARCH_API_KEY")  # Optional for job search

genai.configure(api_key=GOOGLE_API_KEY)

# Load Gemini model
model = genai.GenerativeModel("models/gemini-1.5-pro-002")

# PDF Resume Upload
uploaded_text = ""
uploaded_file = st.file_uploader("Upload Resume (PDF optional)", type="pdf")
if uploaded_file:
    reader = PdfReader(uploaded_file)
    uploaded_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Fallback resume content
sample_resume = uploaded_text or "Experienced professional with a background in strategy, marketing, and AI-driven content development."
if not uploaded_text:
    st.warning("No resume uploaded — using sample resume data for vector analysis.")

# Dynamically create vectorstore
@st.cache_resource
def create_vectorstore():
    docs = [Document(page_content=sample_resume)]
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    return FAISS.from_documents(texts, embeddings)

vectorstore = create_vectorstore()
retriever = vectorstore.as_retriever()

# Streamlit App
st.set_page_config(page_title="Pathfinder AI", page_icon="🧭")
st.title("🧭 Pathfinder AI: Career Coach")
st.markdown("Get resume summaries + job fit insights powered by Gemini.")

# Input Form
with st.form("input_form"):
    name = st.text_input("Name")
    role = st.text_input("Current Role")
    skills = st.text_input("Key Skills")
    goal = st.text_input("Career Goal")
    tone = st.selectbox("Tone", ["confident", "energetic", "professional", "creative"], index=0)
    query = st.text_input("Ask Pathfinder", "")
    location = st.text_input("Job Search Location (optional)")
    submit = st.form_submit_button("Generate")

if submit:
    with st.spinner("Generating resume and retrieving job insights..."):
        # Generate summary
        prompt = f"""
        Write a {tone} professional resume summary for {name}, currently a {role}, \
        with skills in {skills}, seeking a role in {goal}.
        """
        response = model.generate_content(prompt)
        summary = response.text.strip()

        # Query the retriever
        docs = retriever.invoke(query)
        insights = "\n\n".join([doc.page_content for doc in docs])

        # Display results
        st.subheader("Generated Resume Summary")
        st.success(summary)

        st.subheader("Matching Resume Patterns")
        st.info(insights if insights else "No relevant patterns found.")

        # Real-time job search (optional)
        if JSEARCH_API_KEY and goal:
            st.subheader("🔎 Real-Time Job Listings")
            job_api_url = "https://jsearch.p.rapidapi.com/search"
            headers = {
                "X-RapidAPI-Key": JSEARCH_API_KEY,
                "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
            }
            params = {"query": goal, "page": "1"}
            if location:
                params["location"] = location

            try:
                response = requests.get(job_api_url, headers=headers, params=params)
                if response.status_code == 200:
                    results = response.json().get("data", [])
                    if results:
                        for job in results[:5]:
                            st.markdown(f"**{job['job_title']}** at *{job['employer_name']}*  ")
                            st.caption(f"{job['job_city']}, {job['job_state']} | {job['job_employment_type']}")
                            st.write(job['job_description'][:250] + "...")
                            st.markdown(f"[View Job Posting]({job['job_apply_link']})")
                            st.markdown("---")
                    else:
                        st.info("No job matches found for your role yet — try another search.")
                else:
                    st.warning("Unable to fetch job listings. Please try again later.")
            except Exception as e:
                st.error(f"Job search failed: {str(e)}")
        else:
            st.caption("⚠️ Job search API not configured. Add your JSEARCH_API_KEY to enable this feature.")

st.caption("Created by Alison Morano | Powered by Gemini 1.5 + FAISS + LangChain + JSearch")
