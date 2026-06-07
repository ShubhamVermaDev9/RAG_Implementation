"""
RAG Document Ingestion Pipeline
Loads, splits, embeds, and stores documents in ChromaDB for retrieval.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
DEFAULT_DOCS_PATH = "docs"
DEFAULT_DB_PATH = "db/chroma_db"
HASH_FILE = "db/docs_hash.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_docs_hash(docs_path: str) -> str:
    """
    Return an MD5 hex digest of all .txt files in *docs_path*.
    Changing, adding, or removing any file changes the hash.
    """
    h = hashlib.md5()
    root = Path(docs_path)
    for fpath in sorted(root.glob("*.txt")):
        h.update(fpath.name.encode())          # include filename
        h.update(fpath.read_bytes())           # include content
    return h.hexdigest()


def load_saved_hash(hash_file: str = HASH_FILE) -> Optional[str]:
    """Return the previously saved hash, or None if it doesn't exist."""
    path = Path(hash_file)
    if path.exists():
        return json.loads(path.read_text()).get("hash")
    return None


def save_hash(digest: str, hash_file: str = HASH_FILE) -> None:
    """Persist *digest* to *hash_file*."""
    Path(hash_file).parent.mkdir(parents=True, exist_ok=True)
    Path(hash_file).write_text(json.dumps({"hash": digest}, indent=2))


def is_store_stale(docs_path: str, hash_file: str = HASH_FILE) -> bool:
    """
    Return True when the docs have changed since the last ingest,
    or when no hash has been saved yet.
    """
    current = compute_docs_hash(docs_path)
    saved = load_saved_hash(hash_file)
    if saved is None:
        logger.info("No previous hash found — treating store as stale.")
        return True
    if current != saved:
        logger.info("Document hash changed — store is stale.")
        return True
    logger.info("Document hash matches — store is up-to-date.")
    return False


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_documents(docs_path: str = DEFAULT_DOCS_PATH):
    """Load all .txt files from *docs_path*."""
    logger.info("Loading documents from '%s'…", docs_path)

    root = Path(docs_path)
    if not root.exists():
        raise FileNotFoundError(
            f"Directory '{docs_path}' does not exist. "
            "Please create it and add your .txt files."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},  # explicit encoding
        show_progress=True,
    )
    documents = loader.load()

    if not documents:
        raise FileNotFoundError(
            f"No .txt files found in '{docs_path}'. "
            "Please add documents and re-run."
        )

    logger.info("Loaded %d document(s).", len(documents))
    for i, doc in enumerate(documents[:3]):
        logger.info(
            "  [%d] %s — %d chars",
            i + 1,
            doc.metadata.get("source", "unknown"),
            len(doc.page_content),
        )
    return documents


def split_documents(documents, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter,
    which respects paragraph / sentence / word boundaries before hard-cutting.
    """
    logger.info(
        "Splitting documents (chunk_size=%d, overlap=%d)…",
        chunk_size,
        chunk_overlap,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],   # graceful fallback chain
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Enrich metadata so every chunk is self-describing
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_total"] = len(chunks)
        chunk.metadata["chunk_length"] = len(chunk.page_content)

    logger.info("Created %d chunk(s) from %d document(s).", len(chunks), len(documents))

    # Debug preview
    for i, chunk in enumerate(chunks[:5]):
        logger.debug(
            "\n--- Chunk %d ---\nSource : %s\nLength : %d chars\nPreview: %s\n%s",
            i + 1,
            chunk.metadata.get("source"),
            chunk.metadata["chunk_length"],
            chunk.page_content[:200],
            "-" * 50,
        )

    return chunks


def build_embedding_model(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Instantiate the HuggingFace embedding model."""
    logger.info("Loading embedding model '%s'…", model_name)
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_store(
    chunks,
    embedding_model: HuggingFaceEmbeddings,
    persist_directory: str = DEFAULT_DB_PATH,
) -> Chroma:
    """Embed *chunks* and persist them in a ChromaDB collection."""
    logger.info("Creating vector store at '%s'…", persist_directory)
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    count = len(vectorstore.get()["ids"])
    logger.info("Vector store created — %d vectors stored.", count)
    return vectorstore


def load_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    persist_directory: str = DEFAULT_DB_PATH,
) -> Chroma:
    """Load an existing ChromaDB collection from disk."""
    logger.info("Loading existing vector store from '%s'…", persist_directory)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    count = len(vectorstore.get()["ids"])
    logger.info("Loaded vector store — %d vectors found.", count)
    return vectorstore


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_ingestion(
    docs_path: str = DEFAULT_DOCS_PATH,
    db_path: str = DEFAULT_DB_PATH,
    hash_file: str = HASH_FILE,
    force: bool = False,
) -> Chroma:
    """
    Full ingestion pipeline with staleness detection.

    Parameters
    ----------
    docs_path : source directory containing .txt files
    db_path   : ChromaDB persist directory
    hash_file : path to the JSON file storing the last ingest hash
    force     : skip staleness check and always re-ingest

    Returns
    -------
    A ready-to-query Chroma vectorstore instance.
    """
    logger.info("=== RAG Document Ingestion Pipeline ===")

    embedding_model = build_embedding_model()

    store_exists = Path(db_path).exists()

    if store_exists and not force and not is_store_stale(docs_path, hash_file):
        # Fast path — nothing changed
        return load_vector_store(embedding_model, db_path)

    if store_exists and force:
        logger.warning("Force flag set — re-ingesting despite existing store.")

    # Full ingest
    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks, embedding_model, db_path)

    # Persist the new hash so we can detect future changes
    new_hash = compute_docs_hash(docs_path)
    save_hash(new_hash, hash_file)
    logger.info("Hash saved to '%s'.", hash_file)

    logger.info("✅ Ingestion complete — documents are ready for RAG queries.")
    return vectorstore


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG ingestion pipeline")
    parser.add_argument("--docs",  default=DEFAULT_DOCS_PATH, help="Path to documents directory")
    parser.add_argument("--db",    default=DEFAULT_DB_PATH,   help="ChromaDB persist directory")
    parser.add_argument("--force", action="store_true",        help="Force re-ingest even if store is fresh")
    args = parser.parse_args()

    run_ingestion(docs_path=args.docs, db_path=args.db, force=args.force)