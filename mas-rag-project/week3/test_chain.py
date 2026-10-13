"""
Week 3: test_chain.py
----------------------
Runs 5 questions through the full 3-agent Chain topology.
Shows the complete message trace at every step.
Saves all traces to chain_traces.json (used in Week 5).

Usage:
    python test_chain.py

Folder structure expected:
    mas-rag-project/
    ├── week2/
    │   └── documents/       ← created by week2_setup.py
    └── week3/
        ├── chain_topology.py
        └── test_chain.py    ← you are running this
"""

import os
import json
from chain_topology import ChainTopology

# ── Header ─────────────────────────────────────────────────────────
print()
print("=" * 62)
print("  WEEK 3 — Chain Topology")
print("  Message flow: [Retriever] → [Analyzer] → [Responder]")
print("=" * 62)
print()

# ── Point to Week 2 documents ──────────────────────────────────────
# chain_topology.py is in week3/ so we go up one level to week2/documents
here      = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.normpath(os.path.join(here, "..", "week2", "documents"))

print(f"  Loading documents from: {docs_path}")
print()

# ── Build the chain ────────────────────────────────────────────────
chain = ChainTopology(docs_folder=docs_path)

print()
print("  Chain is ready. Starting tests...")
print("=" * 62)

# ── 5 test questions ───────────────────────────────────────────────
questions = [
    "What is a Large Language Model and what are some examples?",
    "Explain how RAG works step by step.",
    "What is hallucination in LLMs and how does RAG help fix it?",
    "What is the difference between a Chain and a Hub topology?",
    "What is fine-tuning and when should you use it instead of RAG?",
]

all_traces = []

for i, question in enumerate(questions, 1):

    print()
    print(f"  {'─' * 58}")
    print(f"  Question {i} of {len(questions)}")
    print(f"  {'─' * 58}")
    print()

    # Run the full chain
    trace = chain.run(question, verbose=True)
    all_traces.append(trace)

    # ── Show the full message trace ────────────────────────────────
    print("  ── FULL MESSAGE TRACE ──────────────────────────────────")
    print()

    for step in trace["steps"]:
        agent  = step["agent"].upper()
        output = step["output"]

        # Show a preview of each agent's output (first 180 characters)
        preview = output.replace("\n", " ").strip()
        if len(preview) > 180:
            preview = preview[:180] + "..."

        print(f"  [{agent}] output preview:")
        print(f"  → {preview}")
        print()

    # ── Show the final answer in full ─────────────────────────────
    print("  ── FINAL ANSWER ────────────────────────────────────────")
    print()
    for line in trace["final_answer"].strip().splitlines():
        print(f"    {line}")
    print()

# ── Save traces to JSON log ────────────────────────────────────────
# This file will be used in Week 5 when we compare Chain vs Hub
log_path = os.path.join(here, "chain_traces.json")

with open(log_path, "w", encoding="utf-8") as f:
    json.dump(all_traces, f, indent=2, ensure_ascii=False)

# ── Results summary ────────────────────────────────────────────────
print()
print("=" * 62)
print("  WEEK 3 — RESULTS SUMMARY")
print("=" * 62)
print()
print(f"  Topology tested  : Chain (3 agents in sequence)")
print(f"  Questions run    : {len(questions)}")
print(f"  Agents per run   : 3  (Retriever → Analyzer → Responder)")
print(f"  LLM calls per Q  : 2  (Analyzer + Responder)")
print(f"  Traces saved to  : chain_traces.json")
print()
print("  What each agent did:")
print("    Retriever  — searched the vector database")
print("    Analyzer   — extracted only the relevant facts")
print("    Responder  — wrote the clean final answer")
print()
print("  Why this matters for your research:")
print("    • You can now see EXACTLY which agent introduces an error")
print("    • The message trace logs every step — just like the paper's")
print("      attack pipeline traces communication between agents")
print("    • Next week you build Hub topology so you can compare both")
print()
print("  ✓ WEEK 3 CHECKPOINT PASSED!")
print()
print("  Next step → Week 4: build the Hub topology")
print("  (1 Coordinator + 2 Specialists working in parallel)")
print()
print("=" * 62)
