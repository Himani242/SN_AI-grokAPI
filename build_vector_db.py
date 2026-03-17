import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DOCUMENT_FOLDER = "documents"
VECTOR_DB_FOLDER = "vector_db"

docs = []

print("Loading documents...")

# walk through all files including subfolders
for root, dirs, files in os.walk(DOCUMENT_FOLDER):

    for file in files:

        path = os.path.join(root, file)

        try:

            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                loaded_docs = loader.load()

            elif file.endswith(".docx"):
                loader = Docx2txtLoader(path)
                loaded_docs = loader.load()

            elif file.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
                loaded_docs = loader.load()

            else:
                continue

            # attach source metadata
            for doc in loaded_docs:
                doc.metadata["source"] = file

            docs.extend(loaded_docs)

        except Exception as e:
            print(f"Error loading {file}: {e}")

print(f"Total documents loaded: {len(docs)}")

# split documents
print("Splitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(docs)

print(f"Total chunks created: {len(chunks)}")

# embedding model
print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# create vector database
print("Creating vector database...")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=VECTOR_DB_FOLDER,
    collection_name="smartnode_docs"
)

vector_db.persist()

print("Vector database created successfully.")
print(f"Saved in folder: {VECTOR_DB_FOLDER}")