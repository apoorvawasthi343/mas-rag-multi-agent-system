import chromadb
import ollama

print("=" * 50)
print("  WEEK 1 - TEST 2: Your First RAG Pipeline!")
print("=" * 50)
print()

# ── STEP 1: Create a local vector database ────────────────────────
print("Step 1: Creating your vector database...")

client = chromadb.Client()
collection = client.create_collection("week1_test")

print("✓ Vector database created")

# ── STEP 2: Load documents into it ────────────────────────────────
print()
print("Step 2: Loading documents into the database...")

documents = [
    "The Hub topology has one central coordinator agent connected "
    "to multiple specialist agents. The coordinator sends the question "
    "to all specialists and combines their answers into one final response.",

    "The Chain topology passes information sequentially. Agent 1 "
    "retrieves documents, then passes its findings to Agent 2 who "
    "analyzes them, then Agent 2 passes to Agent 3 who writes the "
    "final answer.",

    "RAG stands for Retrieval-Augmented Generation. Instead of relying "
    "on memory, the LLM first searches through real documents to find "
    "relevant facts, then uses those facts to write a grounded answer.",

    "A vector database stores text as mathematical numbers called "
    "embeddings. This lets it find documents that are similar in meaning "
    "even when the exact words are different.",

    "Multi-agent systems use several AI agents that each have a specific "
    "role. They communicate by passing messages to each other to solve "
    "tasks that are too complex for one agent alone.",
]

collection.add(
    documents=documents,
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
)

print(f"✓ Loaded {len(documents)} documents into the database")

# ── STEP 3: Search for relevant documents ─────────────────────────
print()
question = "What is a Hub topology and how does it work?"
print("Step 3: Searching for documents relevant to:")
print(f'        "{question}"')

results = collection.query(query_texts=[question], n_results=2)
retrieved_docs = results['documents'][0]

print(f"✓ Found {len(retrieved_docs)} relevant documents:")
for i, doc in enumerate(retrieved_docs, 1):
    preview = doc[:80] + "..." if len(doc) > 80 else doc
    print(f"   [{i}] {preview}")

# ── STEP 4: Send context + question to LLM ────────────────────────
print()
print("Step 4: Sending retrieved context to the LLM...")
print("        (may take 10-20 seconds)")

context = "\n".join(retrieved_docs)

response = ollama.chat(
    model='llama3.2',
    messages=[
        {
            'role': 'system',
            'content': (
                'You are a helpful assistant. '
                'Answer the user question using ONLY the provided context. '
                'Be concise and clear.'
            )
        },
        {
            'role': 'user',
            'content': f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
)

# ── RESULT ────────────────────────────────────────────────────────
print()
print("-" * 50)
print(f"Question: {question}")
print()
print("Answer from LLM:")
print(response['message']['content'])
print("-" * 50)
print()
print("=" * 50)
print("✓ SUCCESS! Your full RAG pipeline is working!")
print()
print("✓ What just happened in 4 steps:")
print("   1. You stored 5 documents in a vector database")
print("   2. You searched it with a natural language question")
print("   3. It found the 2 most relevant documents")
print("   4. The LLM answered using those REAL retrieved facts")
print("   --> That is EXACTLY what RAG is!")
print()
print("✓ WEEK 1 COMPLETE!")
print("  You now have a working LLM + RAG system on your laptop.")
print("  Tell your professor: environment is ready, Week 1 done.")
print("=" * 50)
