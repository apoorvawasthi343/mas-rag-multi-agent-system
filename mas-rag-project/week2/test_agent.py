"""
Week 2: test_agent.py
----------------------
Creates your RAG agent and runs 5 test questions through it.
At the end it prints your accuracy score.

Usage:
    python test_agent.py

Make sure you have run week2_setup.py first!
"""

import os
from single_agent import RAGAgent

# ── Header ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  WEEK 2 — Testing Your AI Knowledge Base Agent")
print("=" * 60)
print()

# ── Create the agent ───────────────────────────────────────────────
docs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")

agent = RAGAgent(
    name="Alex",
    role="an AI research assistant specialising in Large Language Models and AI systems",
    docs_folder=docs_path,
)

print()
print(f"  Agent: {agent}")
print()
print("=" * 60)

# ── 5 test questions ───────────────────────────────────────────────
# Each question has keywords the correct answer MUST mention.
# The script auto-grades by checking if those keywords appear.

test_cases = [
    {
        "question" : "What does RAG stand for and what problem does it solve?",
        "keywords" : ["retrieval", "augmented", "generation", "hallucin"],
        "hint"     : "Should mention: full name + hallucination / knowledge cutoff",
    },
    {
        "question" : "How many parameters does GPT-3 have?",
        "keywords" : ["175", "billion"],
        "hint"     : "Should mention: 175 billion parameters",
    },
    {
        "question" : "What is the difference between fine-tuning and RAG?",
        "keywords" : ["rag", "fine"],
        "hint"     : "Should contrast: RAG = updates knowledge, fine-tuning = changes behaviour",
    },
    {
        "question" : "What is hallucination in LLMs and give two ways to reduce it?",
        "keywords" : ["hallucin", "rag", "temperature"],
        "hint"     : "Should mention: wrong info + two methods (RAG, temperature, citations, etc.)",
    },
    {
        "question" : "Approximately how many words is one token equal to?",
        "keywords" : ["0.75"],
        "hint"     : "Should mention: ~0.75 words per token",
    },
]

# ── Run each question ──────────────────────────────────────────────
results = []

for i, tc in enumerate(test_cases, 1):

    print()
    print(f"  ┌── Question {i} of {len(test_cases)} ──────────────────────────────")
    print(f"  │  {tc['question']}")
    print(f"  └─────────────────────────────────────────────────────")
    print()

    # Get the agent's answer
    answer, sources = agent.answer(tc["question"])

    print("  Agent's answer:")
    print()
    for line in answer.strip().splitlines():
        print(f"    {line}")
    print()

    # Auto-grade: check if key facts are in the answer
    answer_lower  = answer.lower()
    hits          = [kw for kw in tc["keywords"] if kw.lower() in answer_lower]
    passed        = len(hits) >= max(1, len(tc["keywords"]) // 2 + 1)
    results.append(passed)

    status = "✓ PASS" if passed else "✗ NEEDS REVIEW"
    print(f"  Grade : {status}")
    print(f"  Hint  : {tc['hint']}")
    print()

    # Show which document was used (first 120 chars)
    if sources:
        preview = sources[0][:120].replace("\n", " ") + "..."
        print(f"  Source used: \"{preview}\"")

    print()
    print("  " + "─" * 56)

# ── Final score ────────────────────────────────────────────────────
print()
print("=" * 60)
print("  WEEK 2 — RESULTS")
print("=" * 60)
print()

correct  = sum(results)
total    = len(results)
accuracy = (correct / total) * 100

for i, (tc, passed) in enumerate(zip(test_cases, results), 1):
    icon = "✓" if passed else "✗"
    print(f"  {icon}  Q{i}: {tc['question'][:50]}...")

print()
print(f"  Score: {correct}/{total} correct  ({accuracy:.0f}%)")
print()

if accuracy >= 80:
    print("  ✓ WEEK 2 CHECKPOINT PASSED!")
    print("  Your agent answers from real documents, not guesswork.")
    print("  You understand RAG well enough to explain it in an interview.")
    print()
    print("  What you built today:")
    print("    • A named AI agent with a role")
    print("    • A 10-document vector knowledge base")
    print("    • A search-then-answer RAG pipeline")
    print()
    print("  Next step → Week 3: build the Chain topology (3 agents in a line)")
elif accuracy >= 60:
    print("  ⚠️  MOSTLY THERE — a couple of answers need attention.")
    print("  Check the questions marked ✗ above.")
    print("  Read the relevant document in your /documents folder")
    print("  and see if the agent retrieved the right source.")
else:
    print("  ✗ Something needs fixing.")
    print("  Most likely cause: documents were not loaded correctly.")
    print("  Try running:  python week2_setup.py  again,")
    print("  then re-run:  python test_agent.py")

print()
print("=" * 60)
