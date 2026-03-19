import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# load API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# load vector database
vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model,
    collection_name="smartnode_docs"
)

# fetch 20 chunks for better comparison queries
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 20,
        "fetch_k": 50,
        "lambda_mult": 0.7
    }
)

# Groq model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.1
)


def ask_ai(question):

    try:

        docs = retriever.invoke(question)

        if not docs:
            return "I could not find any relevant information in the documents."

        context = ""
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            context += f"\n[Source: {source}]\n{doc.page_content}\n"

        prompt = f"""
You are a strict assistant for Smart Node company.

Your job is to answer questions ONLY using the context provided below.

STRICT RULES:
- If the answer is found in the context, answer clearly and accurately.
- If the answer is NOT in the context, reply exactly: "I don't have information about this in the provided documents."
- Do NOT use your own knowledge.
- Do NOT guess or make up answers.
- Do NOT answer from outside the context.
- For comparison questions, compare point by point in detail covering ALL of these aspects:
  1. Product Overview and Control Method
  2. Supported Model Numbers
  3. Load Terminal Configuration for each model
  4. All Features
  5. Technical Specifications
  6. Dimensions
  7. Safety and Warnings
- Never say a product does not have something unless the context clearly confirms it.
- Always give specific model numbers, values, and details - never give vague general answers.

Context:
{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"ERROR: {str(e)}"
