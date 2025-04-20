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

industry_options = sorted([
    "", "Artificial Intelligence", "Campaigns", "Consulting", "Construction", "Customer Service",
    "Data Science", "Design", "Education", "Engineering", "Finance", "Healthcare", "Hospitality",
    "Human Resources", "Legal", "Manufacturing", "Marketing", "Non-Profit", "Operations",
    "Politics", "Project Management", "Public Relations", "Real Estate", "Retail", "Sales",
    "Supply Chain", "Technology", "XR / Immersive Tech", "Other"
])

with st.form("input_form"):
    name = st.text_input("Name")
    role = st.text_input("Current Role")
    goal = st.text_input("Career Goal")
    tone = st.selectbox(
        "Choose a tone for your resume summary",
        ["professional", "confident", "enthusiastic", "strategic", "creative"]
    )
    location = st.text_input("Job Search Location (optional)")
    industry = st.selectbox("Select Your Industry (optional)", industry_options)

    if industry == "Other":
        custom_industry = st.text_input("Please enter your industry")
        industry_to_use = custom_industry.strip()
    else:
        industry_to_use = industry

    st.caption(
        "🧠 Optional: Select your industry to take a short AI-generated skills quiz. "
        "Check off skills you have — we’ll tailor your resume and suggest free learning resources for any you don’t."
    )
    st.caption("🔁 Be sure to click 'Generate' again anytime you change anything in the form — including your industry or skill selections.")
    submit = st.form_submit_button("Generate")

selected_skills = []
missing_skills = []
generated_skills = []
custom_skills = ""

if submit:
    custom_skills = st.text_input("Additional Skills (optional, comma-separated)")

    if industry_to_use:
        use_quiz = st.checkbox("Take AI-generated skills quiz for this industry?", value=True)

        if use_quiz:
            with st.spinner(f"Loading skill assessment for {industry_to_use}..."):
                skill_prompt = f"List 5 essential skills that professionals in the {industry_to_use} industry should have. Respond with only the list, no explanation."
                skill_response = model.generate_content(skill_prompt)
                st.code(skill_response.text, language="markdown")
                raw_skills = skill_response.text.strip().split("\\n")
            generated_skills = [skill.strip("-• ") for skill in raw_skills if skill.strip()]
            generated_skills = [skill.strip("-• ") for skill in skill_response.text.strip().split("") if skill.strip()]

            st.subheader(f"{industry_to_use} Skill Check")
            
            missing_skills = [s for s in generated_skills if s not in selected_skills]

            if missing_skills:
                st.subheader("📚 Recommended Learning for Skills You Didn't Check")
                for skill in missing_skills:
                    link_prompt = f"Suggest a free or well-known online course or video tutorial for someone who wants to learn '{skill}'. Return only one recommendation."
                    link_response = model.generate_content(link_prompt)
                    st.markdown(f"**{skill}**: {link_response.text.strip()}")

            with st.spinner("Generating resume and retrieving job insights..."):
                combined_skills = ", ".join(selected_skills)
                if custom_skills:
                    combined_skills += ", " + custom_skills

                prompt = f"""
                Write a {tone} professional resume summary for {name}, currently a {role}, \
                with skills in {combined_skills}, seeking a role in {goal}.
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
                            job_count = st.radio("How many job listings would you like to see?", [5, 10, 15], index=0)
                            for job in results[:job_count]:
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
