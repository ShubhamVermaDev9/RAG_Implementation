"""
RAG Query Pipeline
Retrieves relevant chunks from ChromaDB and generates an answer via an LLM.
"""

import os
import sys
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH = "db/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # must match ingest.py
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5
SCORE_THRESHOLD = 0.5   # cosine similarity — 0.0 (no match) → 1.0 (identical)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_vector_store(db_path: str = DEFAULT_DB_PATH) -> Chroma:
    """Load the ChromaDB vector store from disk."""
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Vector store not found at '{db_path}'. "
            "Run ingest.py first to build it."
        )

    logger.info("Loading vector store from '%s'…", db_path)

    # ⚠️  Must use the same embedding model that was used during ingestion.
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    db = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    count = len(db.get()["ids"])
    logger.info("Vector store loaded — %d vectors available.", count)
    return db


def build_retriever(db: Chroma, k: int = TOP_K, score_threshold: float = SCORE_THRESHOLD):
    """
    Return a retriever that filters by cosine similarity score.
    Raise the threshold to get fewer but more relevant chunks.
    Lower it (or remove) to cast a wider net.
    """
    return db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": score_threshold,
        },
    )


def format_docs(docs) -> str:
    """Concatenate retrieved chunks into a single context string."""
    if not docs:
        return "No relevant context found."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        parts.append(
            f"[Source {i}: {Path(source).name}, chunk {chunk_idx}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def build_rag_chain(retriever, llm):
    """
    Assemble the full RAG chain:
        question → retrieve → format → prompt → LLM → parse
    """
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def show_retrieved_docs(docs: list) -> None:
    """Pretty-print retrieved chunks for debugging."""
    if not docs:
        print("\n⚠️  No chunks met the similarity threshold.\n")
        return

    print(f"\n{'=' * 60}")
    print(f"  Retrieved {len(docs)} chunk(s)")
    print(f"{'=' * 60}")
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        chunk_idx = doc.metadata.get("chunk_index", "?")
        length = doc.metadata.get("chunk_length", len(doc.page_content))
        print(f"\n[Chunk {i}]  {source}  •  index={chunk_idx}  •  {length} chars")
        print("-" * 60)
        print(doc.page_content[:400])
        if len(doc.page_content) > 400:
            print(f"  … ({len(doc.page_content) - 400} more chars)")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main query function
# ---------------------------------------------------------------------------

def query(
    question: str,
    db_path: str = DEFAULT_DB_PATH,
    k: int = TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    show_context: bool = True,
) -> str:
    """
    Run a full RAG query: retrieve → format context → generate answer.

    Parameters
    ----------
    question        : the user's question
    db_path         : path to the ChromaDB persist directory
    k               : max number of chunks to retrieve
    score_threshold : minimum cosine similarity (0–1) for a chunk to be included
    show_context    : print retrieved chunks before the answer

    Returns
    -------
    The LLM-generated answer string.
    """
    db = load_vector_store(db_path)
    retriever = build_retriever(db, k=k, score_threshold=score_threshold)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = build_rag_chain(retriever, llm)

    # Optionally show retrieved chunks for transparency / debugging
    if show_context:
        docs = retriever.invoke(question)
        show_retrieved_docs(docs)

    logger.info("Generating answer…")
    answer = chain.invoke(question)
    return answer


# ---------------------------------------------------------------------------
# Synthetic test questions
# ---------------------------------------------------------------------------
SYNTHETIC_QUESTIONS = [
    "What was NVIDIA's first graphics accelerator called?",
    "Which company did NVIDIA acquire to enter the mobile processor market?",
    "What was Microsoft's first hardware product release?",
    "How much did Microsoft pay to acquire GitHub?",
    "In what year did Tesla begin production of the Roadster?",
    "Who succeeded Ze'ev Drori as CEO in October 2008?",
    "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?",
    "What was the original name of Microsoft before it became Microsoft?",
]


def run_batch(db_path: str = DEFAULT_DB_PATH) -> None:
    """Run all synthetic questions and print answers."""
    print("\n=== Batch RAG Evaluation ===\n")
    db = load_vector_store(db_path)
    retriever = build_retriever(db)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = build_rag_chain(retriever, llm)

    for i, q in enumerate(SYNTHETIC_QUESTIONS, 1):
        print(f"Q{i}: {q}")
        answer = chain.invoke(q)
        print(f"A{i}: {answer}\n{'-' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG query pipeline")
    parser.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--db",        default=DEFAULT_DB_PATH, help="ChromaDB directory")
    parser.add_argument("--k",         type=int,   default=TOP_K,            help="Number of chunks to retrieve")
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD,  help="Minimum similarity score (0-1)")
    parser.add_argument("--batch",     action="store_true",                   help="Run all synthetic test questions")
    parser.add_argument("--no-context", action="store_true",                  help="Hide retrieved chunks")
    args = parser.parse_args()

    if args.batch:
        run_batch(db_path=args.db)
        sys.exit(0)

    if args.question:
        # Single question from CLI argument
        answer = query(
            args.question,
            db_path=args.db,
            k=args.k,
            score_threshold=args.threshold,
            show_context=not args.no_context,
        )
        print(f"\n💬 Answer:\n{answer}\n")

    else:
        # Interactive REPL mode
        print("\n=== RAG Query REPL  (type 'exit' to quit) ===\n")
        db = load_vector_store(args.db)
        retriever = build_retriever(db, k=args.k, score_threshold=args.threshold)
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        chain = build_rag_chain(retriever, llm)

        while True:
            try:
                question = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break

            if question.lower() in {"exit", "quit", "q"}:
                print("Bye!")
                break

            if not question:
                continue

            docs = retriever.invoke(question)
            show_retrieved_docs(docs)

            answer = chain.invoke(question)
            print(f"\n💬 Answer:\n{answer}\n{'─' * 60}\n")