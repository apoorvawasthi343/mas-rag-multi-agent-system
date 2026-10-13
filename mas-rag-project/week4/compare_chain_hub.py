"""
Week 4: compare_chain_hub.py
-----------------------------
Prints the Chain answer and the Hub answer for each question,
one after the other, so you can read and judge them yourself.

Usage:
    python compare_chain_hub.py

Needs both files to exist:
    ../week3/chain_traces.json   (made by test_chain.py)
    hub_traces.json              (made by test_hub.py)
"""

import os
import json
import textwrap

here       = os.path.dirname(os.path.abspath(__file__))
chain_path = os.path.normpath(os.path.join(here, "..", "week3", "chain_traces.json"))
hub_path   = os.path.join(here, "hub_traces.json")

for label, path, fix in [
    ("Chain traces", chain_path, "go to week3 and run:  python test_chain.py"),
    ("Hub traces",   hub_path,   "run:  python test_hub.py"),
]:
    if not os.path.isfile(path):
        print()
        print(f"  ⚠️  {label} not found at:")
        print(f"      {path}")
        print(f"      Fix: {fix}")
        print()
        raise SystemExit(1)

with open(chain_path, "r", encoding="utf-8") as f:
    chain_traces = {t["question"]: t for t in json.load(f)}
with open(hub_path, "r", encoding="utf-8") as f:
    hub_traces = {t["question"]: t for t in json.load(f)}


def show(title: str, text: str) -> None:
    print(f"  {title}")
    for para in text.strip().splitlines():
        for line in textwrap.wrap(para, width=70) or [""]:
            print(f"    {line}")
    print()


shared = [q for q in hub_traces if q in chain_traces]

print()
print("=" * 62)
print(f"  CHAIN vs HUB — {len(shared)} shared questions")
print("=" * 62)

for i, q in enumerate(shared, 1):
    print()
    print(f"  Question {i}: {q}")
    print(f"  {'─' * 58}")
    print()
    show("CHAIN answer:", chain_traces[q]["final_answer"])
    show("HUB answer:",   hub_traces[q]["final_answer"])

    spec = [s for s in hub_traces[q]["steps"] if s["agent"].startswith("Specialist")]
    knew = ", ".join(s["agent"] for s in spec if not s["reported_no_info"]) or "neither"
    print(f"  (Hub specialists with relevant info: {knew})")

print()
print("=" * 62)
print("  Now YOU judge. For each question, ask:")
print("    1. Is the answer correct?")
print("    2. Is it complete, or is something important missing?")
print("    3. Did either one say something that is NOT in the documents?")
print("  Jot your notes down. This is exactly what Week 5 formalises.")
print("=" * 62)
print()
