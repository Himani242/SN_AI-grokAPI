import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# load API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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

retriever = vector_db.as_retriever(search_kwargs={"k":3})

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


def ask_ai(question):

    try:

        docs = retriever.invoke(question)

        if not docs:
            return "No relevant information found in documents."

        # limit context size to avoid Gemini errors
        context = "\n\n".join([doc.page_content[:400] for doc in docs[:3]])

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

        # show real error in UI
        return f"ERROR: {str(e)}"