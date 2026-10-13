"""
Week 4: hub_topology.py
------------------------
Three agents connected in a Hub shape: one Coordinator in the middle,
two Specialists on the outside.

Message flow:

                      Your Question
                           |
                           v
                 +-------------------+
                 |    COORDINATOR    |   1. Receives the question
                 |       (hub)       |   2. Sends it to BOTH specialists
                 +-------------------+   3. Combines their reports
                    ^             ^
          question  |             |  question
          + report  |             |  + report
                    v             v
        +----------------+   +----------------+
        | SPECIALIST A   |   | SPECIALIST B   |
        | Foundations    |   | Techniques     |
        | (own 5 docs)   |   | (own 5 docs)   |
        +----------------+   +----------------+

How this differs from the Chain (Week 3):
  - Chain  : one Retriever searches ALL 10 documents, then passes
             its findings down a line.
  - Hub    : each Specialist only knows HALF of the documents.
             The Coordinator has to gather and combine their reports.

Document split (matches the first 6 characters of each filename):
  Specialist A "Foundations": doc_01, doc_02, doc_08, doc_09, doc_10
      (what LLMs are, how they work, famous models, hallucination, tokens)
  Specialist B "Techniques" : doc_03, doc_04, doc_05, doc_06, doc_07
      (RAG, vector databases, prompting, fine-tuning, multi-agent systems)

RESEARCH NOTE: because the Chain's Retriever sees all 10 documents but each
Hub specialist sees only 5, the two topologies differ in BOTH message shape
AND who-knows-what. Mention this in your write-up. To test the Hub with full
knowledge for both specialists, create it with partition_docs=False.
"""

import ollama
import chromadb
import os
from datetime import datetime


SPECIALIST_A_DOCS = {"doc_01", "doc_02", "doc_08", "doc_09", "doc_10"}
SPECIALIST_B_DOCS = {"doc_03", "doc_04", "doc_05", "doc_06", "doc_07"}

SPECIALIST_A_FOCUS = (
    "what LLMs are, how they work, famous models, hallucination, and tokens"
)
SPECIALIST_B_FOCUS = (
    "RAG, vector databases, prompt engineering, fine-tuning, "
    "and multi-agent systems"
)

# Specialists reply with this phrase when their documents don't help.
NO_INFO = "NO RELEVANT INFORMATION"


class HubTopology:
    """
    A 3-agent Hub topology: Coordinator + Specialist A + Specialist B.

    Every run produces a full message trace (same format as the Chain),
    so Week 5 can compare the two topologies side by side.
    """

    def __init__(self, docs_folder: str, partition_docs: bool = True):
        """
        Args:
            docs_folder    : path to the folder with the .txt knowledge files
                             (the documents/ folder created in Week 2)
            partition_docs : True  -> each specialist gets its own half of the docs
                             False -> both specialists get all 10 docs
        """
        self._db    = chromadb.Client()
        self._col_a = self._db.get_or_create_collection("hub_specialist_a")
        self._col_b = self._db.get_or_create_collection("hub_specialist_b")
        self.partition_docs = partition_docs

        self._load_documents(docs_folder)

    # -- Setup ---------------------------------------------------------

    def _load_documents(self, folder: str) -> None:
        """Load the .txt files into each specialist's own vector database."""

        if not os.path.isdir(folder):
            print("  ⚠️  Documents folder not found:")
            print(f"      {folder}")
            print("      Make sure week2_setup.py was run first!")
            return

        ids_a, docs_a, ids_b, docs_b = [], [], [], []

        for filename in sorted(os.listdir(folder)):
            if not filename.endswith(".txt"):
                continue

            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                text = f.read().strip()

            doc_id = filename[:-4]
            prefix = filename[:6]

            if not self.partition_docs:
                ids_a.append(doc_id); docs_a.append(text)
                ids_b.append(doc_id); docs_b.append(text)
            elif prefix in SPECIALIST_A_DOCS:
                ids_a.append(doc_id); docs_a.append(text)
            elif prefix in SPECIALIST_B_DOCS:
                ids_b.append(doc_id); docs_b.append(text)
            else:
                print(f"  ⚠️  {filename} is not assigned to a specialist - skipped")

        if docs_a:
            self._col_a.upsert(documents=docs_a, ids=ids_a)
        if docs_b:
            self._col_b.upsert(documents=docs_b, ids=ids_b)

        print(f"  ✓ Specialist A loaded {len(docs_a)} documents")
        print(f"  ✓ Specialist B loaded {len(docs_b)} documents")

    # -- LLM helper ----------------------------------------------------

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

    # -- The agents ----------------------------------------------------

    def _specialist(self, name: str, focus: str, collection, question: str):
        """
        A Specialist: searches ITS OWN documents and reports back.

        Returns:
            (report text, list of retrieved documents)
        """
        n = min(2, collection.count())
        results = collection.query(query_texts=[question], n_results=n)
        docs    = results["documents"][0]
        context = "\n\n---\n\n".join(docs)

        system = (
            f"You are the {name}, a specialist agent. "
            f"Your knowledge covers: {focus}. "
            "You will receive a question and some documents from your own "
            "knowledge base. Report the key facts from those documents that "
            "help answer the question. Use ONLY the documents provided. "
            "Be brief. "
            f"If the documents contain nothing relevant to the question, "
            f"reply with exactly: {NO_INFO}"
        )
        user = f"Question: {question}\n\nYour documents:\n{context}"

        return self._llm(system, user), docs

    def _coordinator_combine(self, question: str, reports: str) -> str:
        """
        The Coordinator, second job: combine the specialists' reports
        into one final answer.
        """
        system = (
            "You are the Coordinator of a team of specialist agents. "
            "You receive written reports from your specialists. "
            "Combine the relevant reports into one clear, concise, "
            "well-structured final answer for the user. "
            "Ignore any report that says it has no relevant information. "
            "Use ONLY the facts in the reports - do not add outside information."
        )
        user = (
            f"Original question: {question}\n\n"
            f"{reports}\n\n"
            "Write the final answer for the user based only on these reports."
        )
        return self._llm(system, user)

    # -- Main run ------------------------------------------------------

    def run(self, question: str, verbose: bool = True) -> dict:
        """
        Pass a question through the Hub and return the full trace.

        Note: the two specialists run one after the other (not literally
        at the same time) because a laptop runs one LLM call at a time.
        Logically, they work in parallel: neither sees the other's report.
        """

        if verbose:
            print(f"  Question: {question}")
            print()

        trace = {
            "question"    : question,
            "topology"    : "hub",
            "partitioned" : self.partition_docs,
            "timestamp"   : datetime.now().isoformat(),
            "steps"       : [],
            "final_answer": "",
        }

        # -- Coordinator: delegate (no LLM call, just routing) ---------
        if verbose:
            print("  [1/4] COORDINATOR  - sending question to both specialists...")

        trace["steps"].append({
            "agent" : "Coordinator",
            "phase" : "delegate",
            "input" : question,
            "output": "Question sent to Specialist A and Specialist B",
        })

        # -- Specialist A ----------------------------------------------
        if verbose:
            print("  [2/4] SPECIALIST A - searching its documents...")

        out_a, docs_a = self._specialist(
            "Specialist A", SPECIALIST_A_FOCUS, self._col_a, question
        )
        no_info_a = NO_INFO in out_a.upper()

        trace["steps"].append({
            "agent"           : "Specialist A",
            "phase"           : "retrieve+answer",
            "input"           : question,
            "output"          : out_a,
            "n_docs_retrieved": len(docs_a),
            "reported_no_info": no_info_a,
        })

        if verbose:
            print("        ✓ " + ("no relevant info" if no_info_a else "report written"))

        # -- Specialist B ----------------------------------------------
        if verbose:
            print("  [3/4] SPECIALIST B - searching its documents...")

        out_b, docs_b = self._specialist(
            "Specialist B", SPECIALIST_B_FOCUS, self._col_b, question
        )
        no_info_b = NO_INFO in out_b.upper()

        trace["steps"].append({
            "agent"           : "Specialist B",
            "phase"           : "retrieve+answer",
            "input"           : question,
            "output"          : out_b,
            "n_docs_retrieved": len(docs_b),
            "reported_no_info": no_info_b,
        })

        if verbose:
            print("        ✓ " + ("no relevant info" if no_info_b else "report written"))

        # -- Coordinator: combine --------------------------------------
        if verbose:
            print("  [4/4] COORDINATOR  - combining reports into final answer...")

        reports = (
            f"[Report from Specialist A]\n{out_a}\n\n"
            f"[Report from Specialist B]\n{out_b}"
        )
        final_answer = self._coordinator_combine(question, reports)

        trace["steps"].append({
            "agent" : "Coordinator",
            "phase" : "combine",
            "input" : reports,
            "output": final_answer,
        })
        trace["final_answer"] = final_answer

        if verbose:
            print("        ✓ Answer ready")
            print()

        return trace
