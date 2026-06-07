# RAG_Implementation
The ingestion pipeline processes all documents, splits them into meaningful chunks, generates embeddings, and stores them in a vector database while displaying real-time logs so you can monitor every step of the process.
🚀 Features

📥 Ingest Documents

Build your knowledge base by simply placing documents inside the docs folder and running the ingestion pipeline.

The system will:

* Load documents automatically
* Split documents into chunks
* Generate embeddings
* Store vectors in ChromaDB
* Display live ingestion logs

⸻

🔍 Query Mode

Ask a single question and receive an AI-generated answer based on your document collection.

Features:

* Semantic document retrieval
* Retrieved chunks displayed alongside answers
* Source transparency
* Fast vector search


💬 Chat Mode

Features:

* Multi-turn conversations
* Conversation history support
* Context-aware responses
* Follow-up question understanding
* Retrieval-Augmented Generation (RAG)


Documents



    │
    ▼
Document Loader
    │
    ▼
Text Splitter
    │
    ▼
Embedding Model
    │
    ▼
Chroma Vector Database
    │
    ▼
Retriever
    │
    ▼
LLM
    │
    ▼
Answer Generation
