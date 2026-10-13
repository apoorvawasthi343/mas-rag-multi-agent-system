"""
Week 2: single_agent.py
------------------------
Your first real AI agent with a name, a role, and a
RAG-powered knowledge base it searches before answering.

This file defines the RAGAgent class.
You do NOT run this file directly — it is imported by test_agent.py.
"""

import ollama
import chromadb
import os


class RAGAgent:
    """
    A single AI agent that answers questions by searching
    through its own document knowledge base using RAG.

    How it works:
        1. On startup  → loads all .txt files into a local vector database
        2. On question → searches the vector database for relevant documents
        3.             → sends those documents + the question to the LLM
        4.             → returns the grounded answer
    """

    def __init__(self, name: str, role: str, docs_folder: str):
        """
        Create a new RAG agent.

        Args:
            name        : The agent's name  (e.g. "Alex")
            role        : What the agent does  (e.g. "an AI research assistant")
            docs_folder : Path to the folder containing .txt knowledge base files
        """
        self.name = name
        self.role = role

        # Each agent gets its own isolated ChromaDB collection
        self._db         = chromadb.Client()
        self._collection = self._db.create_collection(
            name.lower().replace(" ", "_")
        )

        # Load knowledge base on startup
        self._load_documents(docs_folder)

    # ── Private helpers ────────────────────────────────────────────

    def _load_documents(self, folder: str) -> None:
        """Read every .txt file in *folder* and store it in the vector database."""

        if not os.path.isdir(folder):
            print(f"  ⚠️  Folder not found: {folder}")
            print("      Run week2_setup.py first!")
            return

        documents = []
        ids       = []

        for i, filename in enumerate(sorted(os.listdir(folder))):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                documents.append(f.read().strip())
                ids.append(f"doc_{i:02d}")

        if not documents:
            print(f"  ⚠️  No .txt files found in {folder}")
            return

        self._collection.add(documents=documents, ids=ids)

        print(f"  ✓ {self.name} loaded {len(documents)} documents "
              f"into the knowledge base")

    # ── Public interface ───────────────────────────────────────────

    def answer(self, question: str, n_results: int = 2):
        """
        Answer a question using RAG.

        Args:
            question  : The user's question (plain English string)
            n_results : How many documents to retrieve (default 2)

        Returns:
            answer    (str)       : The LLM's grounded answer
            sources   (list[str]) : The retrieved document chunks used
        """

        # ── Step 1: Retrieve relevant documents ───────────────────
        results      = self._collection.query(
            query_texts=[question],
            n_results=n_results
        )
        sources      = results["documents"][0]
        context      = "\n\n---\n\n".join(sources)

        # ── Step 2: Generate answer with the retrieved context ─────
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.name}, {self.role}. "
                        "Answer questions using ONLY the context provided below. "
                        "If the context does not contain enough information to answer, "
                        "say exactly: 'I don't have enough information about that "
                        "in my knowledge base.' "
                        "Be accurate, clear, and concise."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
        )

        answer = response["message"]["content"]
        return answer, sources

    # ── Nice printing ──────────────────────────────────────────────

    def __repr__(self) -> str:
        count = self._collection.count()
        return (
            f"RAGAgent(name='{self.name}', "
            f"role='{self.role}', "
            f"documents={count})"
        )
