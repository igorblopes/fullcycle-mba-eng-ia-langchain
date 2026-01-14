import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()


def ingest_pdf():
    for k in ("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
        if k not in os.environ:
            raise RuntimeError(f"Environment variable {k} not set")
        
    PDF_PATH = os.getenv("PDF_PATH")


    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    store.delete_collection()
    store.create_collection()

    docs = PyPDFLoader(str(PDF_PATH)).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        add_start_index=False).split_documents(docs)
    
    if not splits:
        raise SystemExit(0)

    enriched = [
        Document(
            page_content=doc.page_content,
            metadata={k: v for k, v in doc.metadata.items() if v not in ("", None)}
        )
        for doc in splits
    ]

    ids = [f"doc-{i}" for i in range(len(enriched))]

    store.add_documents(documents=enriched, ids=ids)

if __name__ == "__main__":
    ingest_pdf()