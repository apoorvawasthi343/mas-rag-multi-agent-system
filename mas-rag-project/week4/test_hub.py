"""
Week 4: test_hub.py
--------------------
Runs the SAME 5 questions you used in Week 3 through the Hub topology.
Saves every message to hub_traces.json (used in Week 5).

Usage:
    python test_hub.py

Folder structure expected:
    mas-rag-project/
    ├── week2/
    │   └── documents/
    ├── week3/
    │   └── chain_traces.json
    └── week4/
        ├── hub_topology.py
        └── test_hub.py      <- you are running this
"""

import os
import json
from hub_topology import HubTopology

print()
print("=" * 62)
print("  WEEK 4 — Hub Topology")
print("  Coordinator <-> Specialist A + Specialist B")
print("=" * 62)
print()

here      = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.normpath(os.path.join(here, "..", "week2", "documents"))

print(f"  Loading documents from: {docs_path}")
print()

hub = HubTopology(docs_folder=docs_path)

print()
print("  Hub is ready. Starting tests...")
print("=" * 62)

# IMPORTANT: keep this list identical to test_chain.py,
# so Chain and Hub are compared on exactly the same questions.
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

    trace = hub.run(question, verbose=True)
    all_traces.append(trace)

    # Which specialists actually had something useful?
    spec_steps = [s for s in trace["steps"] if s["agent"].startswith("Specialist")]
    print("  ── WHO KNEW WHAT ───────────────────────────────────────")
    for s in spec_steps:
        verdict = "had NOTHING relevant" if s["reported_no_info"] else "contributed facts"
        print(f"    {s['agent']}: {verdict}")
    print()

    print("  ── FINAL ANSWER ────────────────────────────────────────")
    print()
    for line in trace["final_answer"].strip().splitlines():
        print(f"    {line}")
    print()

# Save traces
log_path = os.path.join(here, "hub_traces.json")
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(all_traces, f, indent=2, ensure_ascii=False)

# Summary
answered = sum(1 for t in all_traces if t["final_answer"].strip())

print()
print("=" * 62)
print("  WEEK 4 — RESULTS SUMMARY")
print("=" * 62)
print()
print(f"  Topology tested  : Hub (Coordinator + 2 Specialists)")
print(f"  Questions run    : {len(questions)}")
print(f"  Questions answered: {answered}/{len(questions)}")
print(f"  LLM calls per Q  : 3  (Specialist A + Specialist B + Coordinator)")
print(f"  Traces saved to  : hub_traces.json")
print()

if answered == len(questions):
    print("  ✓ WEEK 4 CHECKPOINT PASSED!")
    print("  Hub produced an answer for every question.")
    print()
    print("  Next: run  python compare_chain_hub.py  to read Chain and Hub")
    print("  answers side by side. That is the start of your real results.")
else:
    print("  ⚠️  Some questions got no answer. Check hub_traces.json")
    print("  and make sure Ollama is running.")

print()
print("=" * 62)
