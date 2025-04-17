import streamlit as st
import google.generativeai as genai
import os
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load API key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# Load Gemini model
model = genai.GenerativeModel("models/gemini-1.5-pro-002")

# Dynamically create vectorstore
@st.cache_resource
def create_vectorstore():
    sample_resume = "Marketing Specialist with 5 years of experience in SEO, content marketing, and automation. Looking for Digital Marketing Manager role."
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
    name = st.text_input("Name", "Alex Rivera")
    role = st.text_input("Current Role", "Marketing Specialist")
    skills = st.text_input("Key Skills", "SEO, content marketing, email automation")
    goal = st.text_input("Career Goal", "Digital Marketing Manager")
    tone = st.selectbox("Tone", ["confident", "energetic", "professional", "creative"], index=1)
    query = st.text_input("Ask Pathfinder", "Find resume patterns for a mid-level marketing manager with automation experience.")
    submit = st.form_submit_button("Generate")

if submit:
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

st.caption("Created by Alison Morano | Powered by LangGraph, FAISS, Gemini 1.5")
