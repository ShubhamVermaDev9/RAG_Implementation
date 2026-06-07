"""
Conversational RAG Pipeline with Memory
Supports multi-turn chat — the LLM remembers previous exchanges within a session.
"""

import sys
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
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
# Constants — keep in sync with ingest.py
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH    = "db/chroma_db"
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"   # ← must match ingest.py
LLM_MODEL          = "gpt-4o-mini"   # cheaper & fast enough for RAG
TOP_K              = 5
SCORE_THRESHOLD    = 0.5
MAX_HISTORY_TURNS  = 6   # keep last N human+AI pairs to avoid context bloat

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful, concise assistant.
Answer questions using ONLY the context snippets provided.
If the answer isn't in the context, say exactly:
"I don't have enough information to answer that based on the provided documents."
Never fabricate facts."""


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def load_vector_store(db_path: str = DEFAULT_DB_PATH) -> Chroma:
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Vector store not found at '{db_path}'. Run ingest.py first."
        )
    logger.info("Loading vector store from '%s'…", db_path)

    # ⚠️  Must match the embedding model used in ingest.py
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    count = len(db.get()["ids"])
    logger.info("Vector store ready — %d vectors.", count)
    return db


def get_retriever(db: Chroma, k: int = TOP_K, threshold: float = SCORE_THRESHOLD):
    return db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": threshold},
    )


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(docs: list) -> str:
    """Turn retrieved docs into a clearly labelled context block."""
    if not docs:
        return "No relevant context found."
    parts = []
    for i, doc in enumerate(docs, 1):
        source    = Path(doc.metadata.get("source", "unknown")).name
        chunk_idx = doc.metadata.get("chunk_index", "?")
        parts.append(
            f"[{i}] {source} (chunk {chunk_idx})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def show_context(docs: list) -> None:
    """Debug-print retrieved chunks."""
    if not docs:
        print("\n⚠️  No chunks met the similarity threshold.\n")
        return
    print(f"\n{'─' * 60}")
    print(f"  {len(docs)} chunk(s) retrieved")
    print(f"{'─' * 60}")
    for i, doc in enumerate(docs, 1):
        source    = Path(doc.metadata.get("source", "unknown")).name
        chunk_idx = doc.metadata.get("chunk_index", "?")
        print(f"\n[{i}] {source} · chunk {chunk_idx}")
        print(doc.page_content[:350])
        if len(doc.page_content) > 350:
            print(f"    … ({len(doc.page_content) - 350} more chars)")
    print(f"{'─' * 60}\n")


# ---------------------------------------------------------------------------
# Conversational RAG
# ---------------------------------------------------------------------------

class ConversationalRAG:
    """
    Stateful multi-turn RAG.

    Each call to `chat()`:
      1. Retrieves relevant chunks for the NEW question.
      2. Builds a prompt that includes: system prompt + chat history + context + question.
      3. Calls the LLM and appends both sides to history.

    The history is trimmed to MAX_HISTORY_TURNS to avoid ballooning token costs.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        k: int = TOP_K,
        threshold: float = SCORE_THRESHOLD,
        verbose: bool = True,
    ):
        self.db        = load_vector_store(db_path)
        self.retriever = get_retriever(self.db, k=k, threshold=threshold)
        self.llm       = ChatOpenAI(model=LLM_MODEL, temperature=0)
        self.verbose   = verbose
        self.history: list[HumanMessage | AIMessage] = []

        # Prompt: system + rolling history + injected context + current question
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content=(
                "Context from documents:\n{context}\n\n"
                "Question: {question}"
            )),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    # ------------------------------------------------------------------

    def _trim_history(self) -> list:
        """Return the last MAX_HISTORY_TURNS turn-pairs."""
        # Each turn = 1 HumanMessage + 1 AIMessage → 2 items per turn
        keep = MAX_HISTORY_TURNS * 2
        return self.history[-keep:] if len(self.history) > keep else self.history

    # ------------------------------------------------------------------

    def chat(self, question: str) -> str:
        """Ask a question; get a grounded answer; history is updated automatically."""
        # 1. Retrieve
        docs    = self.retriever.invoke(question)
        context = format_context(docs)

        if self.verbose:
            show_context(docs)

        # 2. Generate
        logger.info("Generating answer…")
        answer = self.chain.invoke({
            "history":  self._trim_history(),
            "context":  context,
            "question": question,
        })

        # 3. Update history
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))

        return answer

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()
        logger.info("Conversation history cleared.")

    def print_history(self) -> None:
        """Pretty-print the full conversation so far."""
        if not self.history:
            print("(no history yet)")
            return
        for msg in self.history:
            role = "You" if isinstance(msg, HumanMessage) else "Assistant"
            print(f"\n{role}: {msg.content}")


# ---------------------------------------------------------------------------
# Single-shot helper (for scripting / imports)
# ---------------------------------------------------------------------------

def ask(question: str, db_path: str = DEFAULT_DB_PATH, verbose: bool = True) -> str:
    """One-shot RAG query — no conversation state."""
    rag = ConversationalRAG(db_path=db_path, verbose=verbose)
    return rag.chat(question)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Conversational RAG pipeline")
    parser.add_argument("question",     nargs="?",  help="One-shot question (omit for REPL)")
    parser.add_argument("--db",         default=DEFAULT_DB_PATH)
    parser.add_argument("--k",          type=int,   default=TOP_K)
    parser.add_argument("--threshold",  type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--no-context", action="store_true", help="Hide retrieved chunks")
    args = parser.parse_args()

    verbose = not args.no_context
    rag = ConversationalRAG(db_path=args.db, k=args.k, threshold=args.threshold, verbose=verbose)

    # ── One-shot mode ──────────────────────────────────────────────────
    if args.question:
        answer = rag.chat(args.question)
        print(f"\n💬 Answer:\n{answer}\n")
        sys.exit(0)

    # ── Interactive REPL ───────────────────────────────────────────────
    print("\n=== Conversational RAG  (commands: 'history', 'reset', 'exit') ===\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        if question.lower() == "history":
            rag.print_history()
            continue

        if question.lower() == "reset":
            rag.reset()
            print("🔄 Conversation reset.\n")
            continue

        answer = rag.chat(question)
        print(f"\n💬 {answer}\n{'─' * 60}\n")