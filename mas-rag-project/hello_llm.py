import ollama

print("=" * 50)
print("  WEEK 1 - TEST 1: Your First LLM Call!")
print("=" * 50)
print()
print("Asking the LLM a question...")
print("(This may take 10-20 seconds the first time)")
print()

response = ollama.chat(
    model='llama3.2',
    messages=[
        {
            'role': 'user',
            'content': (
                'What is RAG (Retrieval-Augmented Generation)? '
                'Explain it in 3 short sentences like you are '
                'explaining to a university student.'
            )
        }
    ]
)

print("Question: What is RAG?")
print()
print("Answer:")
print(response['message']['content'])
print()
print("=" * 50)
print("✓ SUCCESS! Your LLM is working perfectly!")
print("✓ Checkpoint 1 DONE — now run hello_rag.py")
print("=" * 50)
