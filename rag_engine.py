import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# load .env file
load_dotenv()

# load API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Groq model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.2
)


def ask_ai(question):

    try:

        docs = retriever.invoke(question)

        if not docs:
            return "No relevant information found in documents."

        context = "\n\n".join([doc.page_content for doc in docs[:3]])

        prompt = f"""
You are an assistant for Smart Node company.

Answer the question using only the context below.

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:
        return f"ERROR: {str(e)}"
