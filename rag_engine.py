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

retriever = vector_db.as_retriever(search_kwargs={"k": 10})

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

        # show which source each chunk came from
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
- For comparison questions, carefully read data from ALL sources before answering.
- Never say a product does not have something unless the context clearly confirms it.

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"ERROR: {str(e)}"
