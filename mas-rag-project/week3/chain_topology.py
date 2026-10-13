"""
Week 3: chain_topology.py
--------------------------
Three agents connected in a straight line — the Chain topology.

Message flow:
    Your Question
         ↓
  ┌─────────────────┐
  │  Agent 1        │  Searches the vector database
  │  RETRIEVER      │  Returns: the 2 most relevant documents
  └────────┬────────┘
           │ (retrieved documents passed as message)
           ↓
  ┌─────────────────┐
  │  Agent 2        │  Reads the documents
  │  ANALYZER       │  Returns: structured key facts only
  └────────┬────────┘
           │ (key facts passed as message)
           ↓
  ┌─────────────────┐
  │  Agent 3        │  Reads the key facts
  │  RESPONDER      │  Returns: the final polished answer
  └─────────────────┘
         ↓
    Final Answer

Why 3 agents instead of 1?
  - Each agent does ONE job really well
  - Errors are traceable: we can see exactly which agent caused a wrong answer
  - This is the Chain topology from the research paper — now YOU built it
"""

import ollama
import chromadb
import os
import json
from datetime import datetime


class ChainTopology:
    """
    A 3-agent Chain topology.

    Agents communicate strictly in sequence:
        Retriever → Analyzer → Responder

    Every run produces a full message trace that is saved
    as a JSON log — important for Week 5 when we compare
    Chain vs Hub results side by side.
    """

    def __init__(self, docs_folder: str):
        """
        Set up the chain and load the knowledge base.

        Args:
            docs_folder : path to the folder with .txt knowledge files
                          (the documents/ folder created in Week 2)
        """
        # All 3 agents share one vector database (the knowledge base)
        self._db         = chromadb.Client()
        self._collection = self._db.create_collection("chain_knowledge_base")

        self._load_documents(docs_folder)

    # ── Setup ──────────────────────────────────────────────────────────

    def _load_documents(self, folder: str) -> None:
        """Load every .txt file from the folder into the vector database."""

        if not os.path.isdir(folder):
            print(f"  ⚠️  Documents folder not found:")
            print(f"      {folder}")
            print("      Make sure week2_setup.py was run first!")
            return

        documents, ids = [], []

        for i, filename in enumerate(sorted(os.listdir(folder))):
            if filename.endswith(".txt"):
                path = os.path.join(folder, filename)
                with open(path, "r", encoding="utf-8") as f:
                    documents.append(f.read().strip())
                    ids.append(f"doc_{i:02d}")

        if not documents:
            print(f"  ⚠️  No .txt files found in: {folder}")
            return

        self._collection.add(documents=documents, ids=ids)
        print(f"  ✓ Knowledge base ready — {len(documents)} documents loaded")

    # ── LLM helper ────────────────────────────────────────────────────

    def _llm(self, system: str, user: str) -> str:
        """Make one LLM call and return the text response."""
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response["message"]["content"]

    # ── The 3 agents ──────────────────────────────────────────────────

    def _retriever(self, question: str) -> tuple:
        """
        Agent 1 — RETRIEVER
        Job    : Search the vector database, return the raw documents.
        Input  : The user's question
        Output : Raw retrieved text (passed to the Analyzer)
        """
        results  = self._collection.query(query_texts=[question], n_results=2)
        docs     = results["documents"][0]

        # Format retrieved documents into a clear message
        message = "RETRIEVED DOCUMENTS:\n\n"
        for i, doc in enumerate(docs, 1):
            message += f"[Document {i}]\n{doc}\n\n"

        return message, docs

    def _analyzer(self, question: str, retriever_output: str) -> str:
        """
        Agent 2 — ANALYZER
        Job    : Read the retrieved documents and extract ONLY the key facts.
        Input  : The retrieved documents message from the Retriever
        Output : Structured key facts (passed to the Responder)
        """
        system = (
            "You are a precise analytical agent. "
            "You receive raw documents retrieved from a knowledge base. "
            "Your ONLY job is to extract and list the key facts that directly "
            "answer the given question. "
            "Do NOT write a full answer — just list the relevant facts clearly. "
            "If a document does not contain relevant information, ignore it."
        )
        user = (
            f"Question to answer: {question}\n\n"
            f"{retriever_output}\n"
            "Extract only the key facts from the documents above "
            "that are relevant to answering the question."
        )
        return self._llm(system, user)

    def _responder(self, question: str, analyzer_output: str) -> str:
        """
        Agent 3 — RESPONDER
        Job    : Turn the Analyzer's key facts into a clean final answer.
        Input  : Structured key facts from the Analyzer
        Output : The final polished answer for the user
        """
        system = (
            "You are a clear and friendly communicator. "
            "You receive a list of key facts extracted from a knowledge base. "
            "Your job is to write a clear, concise, well-structured final answer "
            "for the user. Use the facts provided — do not add any information "
            "that is not in the facts list."
        )
        user = (
            f"Original question: {question}\n\n"
            f"Key facts to use:\n{analyzer_output}\n\n"
            "Write the final answer for the user based only on these facts."
        )
        return self._llm(system, user)

    # ── Main run ──────────────────────────────────────────────────────

    def run(self, question: str, verbose: bool = True) -> dict:
        """
        Pass a question through all 3 agents and return the full trace.

        Args:
            question : The question to answer
            verbose  : Print step-by-step progress (default True)

        Returns:
            trace dict with every agent's input and output,
            plus the final answer.
        """

        if verbose:
            print(f"  Question: {question}")
            print()

        trace = {
            "question"    : question,
            "topology"    : "chain",
            "timestamp"   : datetime.now().isoformat(),
            "steps"       : [],
            "final_answer": "",
        }

        # ── Agent 1: Retriever ────────────────────────────────────
        if verbose:
            print("  [1/3] RETRIEVER  — searching knowledge base...")

        retriever_output, sources = self._retriever(question)

        trace["steps"].append({
            "agent"  : "Retriever",
            "input"  : question,
            "output" : retriever_output,
            "n_docs_retrieved" : len(sources),
        })

        if verbose:
            print(f"        ✓ {len(sources)} document(s) retrieved")

        # ── Agent 2: Analyzer ─────────────────────────────────────
        if verbose:
            print("  [2/3] ANALYZER   — extracting key facts...")

        analyzer_output = self._analyzer(question, retriever_output)

        trace["steps"].append({
            "agent"  : "Analyzer",
            "input"  : retriever_output,
            "output" : analyzer_output,
        })

        if verbose:
            print("        ✓ Key facts extracted")

        # ── Agent 3: Responder ────────────────────────────────────
        if verbose:
            print("  [3/3] RESPONDER  — writing final answer...")

        final_answer = self._responder(question, analyzer_output)

        trace["steps"].append({
            "agent"  : "Responder",
            "input"  : analyzer_output,
            "output" : final_answer,
        })

        trace["final_answer"] = final_answer

        if verbose:
            print("        ✓ Answer ready")
            print()

        return trace
