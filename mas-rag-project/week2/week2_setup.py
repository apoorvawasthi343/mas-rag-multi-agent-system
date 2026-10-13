"""
Week 2 Setup Script
-------------------
Run this ONCE to automatically create your documents folder
and all 10 knowledge base files about AI & LLMs.

Usage:
    python week2_setup.py
"""

import os

# ── Create folder structure ────────────────────────────────────────
week2_dir = os.path.dirname(os.path.abspath(__file__))
docs_dir  = os.path.join(week2_dir, "documents")
os.makedirs(docs_dir, exist_ok=True)

print("=" * 55)
print("  WEEK 2 SETUP — Building your AI & LLMs Knowledge Base")
print("=" * 55)
print()

# ── All 10 documents ──────────────────────────────────────────────
documents = {

"doc_01_what_is_llm.txt": """What is a Large Language Model (LLM)?

A Large Language Model (LLM) is a type of artificial intelligence system trained on massive amounts of text data to understand and generate human language. LLMs are built using deep learning, specifically a neural network architecture called the Transformer.

LLMs are called "large" because they have billions of parameters — numerical values the model learns during training. For example, GPT-3 has 175 billion parameters, and GPT-4 is estimated to have over 1 trillion parameters.

LLMs can perform many tasks without being specifically trained for each one: answering questions, writing essays, summarising text, translating languages, writing code, and solving math problems. This ability to handle diverse tasks from general training is called emergent behaviour.

The most well-known LLMs include OpenAI's GPT-4, Anthropic's Claude, Google's Gemini, and Meta's LLaMA. These models power products like ChatGPT, Claude.ai, and Google Gemini.
""",

"doc_02_how_llms_work.txt": """How Do LLMs Work? The Transformer Architecture

LLMs are built on the Transformer architecture, introduced by Google researchers in a 2017 paper titled "Attention is All You Need." The key innovation was the attention mechanism, which allows the model to weigh the importance of every word relative to every other word when generating output.

Training an LLM happens in two main stages:
1. Pre-training: The model reads hundreds of billions of words from the internet, books, and code. It learns to predict the next word in a sequence. Through this process it learns grammar, facts, reasoning, and coding skills.
2. Fine-tuning: After pre-training, the model is further trained on curated examples of good conversations to make it helpful and safe. This stage is called RLHF — Reinforcement Learning from Human Feedback.

LLMs generate text by predicting the next token (word piece) one at a time. Given a prompt, the model calculates probabilities for every possible next token and samples from them.

The temperature parameter controls randomness. A temperature of 0 makes outputs fully deterministic (always the highest-probability token). Higher temperatures introduce creativity and variety.
""",

"doc_03_what_is_rag.txt": """What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is a technique that improves LLM responses by first searching a knowledge base for relevant documents, then using those documents to generate a grounded answer.

The problem RAG solves: LLMs have a training cutoff date and can hallucinate — making up facts — because they rely on what they memorised during training. RAG fixes this by giving the LLM access to real, current documents at query time.

How RAG works in four steps:
1. Index: Documents are converted into vector embeddings and stored in a vector database.
2. Retrieve: When a user asks a question, the question is also converted to an embedding and the most similar document chunks are retrieved.
3. Augment: The retrieved documents are added to the LLM prompt as context.
4. Generate: The LLM reads the context and writes an answer based on those real documents.

RAG was introduced in a 2020 paper by Facebook AI titled "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."

The main advantage of RAG over fine-tuning is that the knowledge base can be updated without retraining the model — just add new documents to the database.
""",

"doc_04_vector_databases.txt": """Vector Databases and Embeddings

A vector database is a specialised database that stores and searches data as high-dimensional numerical vectors called embeddings. Unlike traditional databases that search for exact keyword matches, vector databases find semantically similar content even when the exact words differ.

An embedding is a list of numbers (a vector) that represents the meaning of a piece of text. For example, the sentences "The dog ran fast" and "The canine sprinted quickly" would have very similar embeddings even though they share no words, because they mean the same thing.

Embeddings are created by embedding models — smaller neural networks trained specifically to convert text into vectors. Popular embedding models include OpenAI text-embedding-ada-002 and Sentence Transformers.

To search a vector database, the query is converted to an embedding, and then the stored vectors closest to it are found using a distance metric called cosine similarity. Closer vectors mean more semantically relevant documents.

Popular vector databases include:
- ChromaDB: Open-source, runs locally, perfect for learning and small projects
- Pinecone: Cloud-hosted, highly scalable for production systems
- FAISS: Facebook's library, very fast for large datasets
- Weaviate: Open-source with many integrations

In a RAG system, the vector database acts as the searchable memory that stores all documents.
""",

"doc_05_prompt_engineering.txt": """Prompt Engineering: Getting the Best from LLMs

Prompt engineering is the practice of carefully designing inputs (prompts) to get better, more accurate, and more useful outputs from LLMs. It is one of the most important practical skills for working with AI systems.

Key prompt engineering techniques:

1. Role prompting: Tell the model what role to play. Example: "You are an expert Python developer. Review this code and find bugs." Giving the LLM a specific role improves consistency and quality.

2. Few-shot prompting: Provide 2-3 examples of the desired input-output format before your actual question. This teaches the model exactly what style or structure you want without any fine-tuning.

3. Chain-of-thought prompting: Add "Let's think step by step" to encourage the model to reason through a problem before giving a final answer. This dramatically improves performance on math and logic tasks.

4. System prompts: In chat-based models, the system prompt sets overall context and constraints. Example: "You are a customer service agent. Only answer questions about our products. Be polite and concise."

5. Constrained output: Ask for responses in a specific format — JSON, bullet points, or numbered lists — to make outputs easier to process programmatically.

The core principle: LLMs are very sensitive to exact wording. Small changes to a prompt can dramatically change the output. Clear, specific, and contextual prompts consistently outperform vague ones.
""",

"doc_06_fine_tuning.txt": """Fine-Tuning Large Language Models

Fine-tuning is the process of further training a pre-trained LLM on a smaller, task-specific dataset to improve its performance on a particular domain or task.

Why fine-tune instead of just using prompts?
- Consistency: Fine-tuned models reliably follow a specific format or tone
- Efficiency: They achieve the same results with shorter prompts, saving cost
- Domain expertise: Fine-tuning on medical, legal, or technical data improves accuracy in those fields
- Privacy: You can train on private data without sending it to an external API

Types of fine-tuning:
1. Full fine-tuning: All model parameters are updated. Most powerful but requires enormous compute resources.
2. LoRA (Low-Rank Adaptation): Only a small set of additional parameters are trained alongside the frozen base model. Much cheaper and nearly as effective. This is currently the most popular method.
3. RLHF (Reinforcement Learning from Human Feedback): The model is fine-tuned using human ratings of its outputs to make it helpful and safe. OpenAI used RLHF to create ChatGPT.

Fine-tuning vs RAG — when to use each:
- Use RAG when you need up-to-date or frequently changing information
- Use fine-tuning when you need to change the model's behaviour, style, or persona
- Many production systems combine both: fine-tune for behaviour, RAG for knowledge
""",

"doc_07_multi_agent_systems.txt": """Multi-Agent AI Systems

A multi-agent system (MAS) is an AI architecture where multiple LLM agents work together, each with a specific role, to complete tasks that are too complex for a single agent.

Each agent in a MAS typically has:
- A role: defined by its system prompt (e.g., Researcher, Coder, Reviewer, Supervisor)
- Tools: access to search engines, code executors, databases, or APIs
- Memory: context from previous messages in the conversation
- Communication: the ability to send and receive messages from other agents

Communication topologies define how agents are connected:
- Chain: Agents work in a strict sequence, each passing results to the next
- Hub: A central coordinator sends tasks to specialist agents and combines their results
- Mesh: All agents can communicate directly with all other agents
- Hierarchy: Agents are organised in a tree structure with managers and workers

Popular multi-agent frameworks include LangGraph, CrewAI, AutoGen, and MetaGPT.

The key advantage of multi-agent systems is specialisation. Just as human teams perform better when each person focuses on their expertise, AI teams perform better when each agent has a specific role.

Security is an important concern in MAS. Because agents communicate by passing messages, a compromised or tampered message can propagate errors through the entire system — a single corrupted channel can affect the final output.
""",

"doc_08_famous_llms.txt": """Famous Large Language Models and Who Made Them

GPT-4 (OpenAI, 2023): OpenAI's flagship model, powering ChatGPT Plus. GPT-4 is multimodal — it accepts both text and images as input. It scored in the top 10 percent on the bar exam and performs at a human level on many professional benchmarks.

Claude (Anthropic, 2023-2024): Developed by Anthropic with a focus on safety and honesty. Claude 3 has a context window of up to 200,000 tokens, allowing it to process book-length documents in one request. The main variants are Opus (most powerful), Sonnet (balanced), and Haiku (fastest).

Gemini (Google DeepMind, 2023): Google's multimodal LLM, integrated into Search, Gmail, and Workspace. Gemini Ultra is the most powerful variant, while Gemini Nano runs on mobile devices on-device.

LLaMA (Meta AI, 2023-2024): Meta's open-source LLM family. LLaMA 3.2 can be downloaded and run locally on consumer hardware using tools like Ollama. Available in 1B, 3B, 11B, and 90B parameter variants. Because it is open-source, it is widely used in research.

Mistral (Mistral AI, 2023): A French AI company's open-source models known for efficiency. Mistral 7B outperforms LLaMA 2 13B on many benchmarks despite being smaller.

Qwen (Alibaba, 2023-2024): Alibaba's open-source LLM family with strong performance in both English and Chinese, and notable coding and math capabilities.
""",

"doc_09_hallucination.txt": """Hallucination in LLMs: What It Is and How to Fix It

Hallucination refers to when an LLM generates information that sounds confident and plausible but is factually wrong or entirely made up. It is one of the most critical challenges in deploying LLMs in real-world applications.

Why do LLMs hallucinate?
LLMs are trained to predict the next most likely token — not to verify facts. They learn statistical patterns in language, not a ground-truth model of the world. When asked about topics with limited training data, they fill gaps with plausible-sounding but incorrect information.

Types of hallucination:
1. Factual hallucination: Making up facts, statistics, dates, or citations — for example, inventing a research paper that does not exist.
2. Reasoning hallucination: Logical errors where the conclusion does not follow from the premises.
3. Context hallucination: Ignoring information provided in the prompt and answering from training memory instead.

How to reduce hallucination:
1. RAG: Ground the model in real retrieved documents so it answers from facts, not memory
2. Lower temperature: Reduces random, creative but inaccurate responses
3. Ask for citations: Prompting the model to cite its sources exposes unsupported claims
4. Verification agents: Use a second LLM to fact-check the first LLM's output
5. Constrained prompting: Instruct the model to say "I don't know" when it lacks information

Hallucination rates vary across models. GPT-4 and Claude hallucinate less than smaller models, but no LLM is completely hallucination-free today.
""",

"doc_10_tokens.txt": """Tokens and Context Windows in LLMs

A token is the basic unit of text that an LLM processes. Tokens are not exactly words — they are word pieces produced by a tokeniser. Common short words are usually one token, but longer or rare words may be split into multiple tokens. For example, "hello" is 1 token, and "unbelievable" might split into "un", "believ", "able" — 3 tokens.

A useful rule of thumb: 1 token is approximately 0.75 words in English, or about 4 characters. The sentence "Large language models process text as tokens" contains roughly 8 tokens.

Why tokens matter for developers:
- LLM APIs charge per token (input tokens + output tokens)
- Context window limits are measured in tokens, not words
- Longer prompts cost more and take more time to process

The context window is the maximum number of tokens an LLM can process in one request — the prompt plus the response combined. Text beyond the context window is simply cut off and the model cannot see it.

Context window sizes by model:
- GPT-3.5: 16,000 tokens (roughly 12,000 words)
- GPT-4: 128,000 tokens (roughly 96,000 words)
- Claude 3: 200,000 tokens (roughly 150,000 words)
- LLaMA 3.2: 128,000 tokens

The attention mechanism in Transformers runs in O(n squared) time relative to the number of tokens, which is why very large context windows are computationally expensive to process.
""",

}

# ── Write all documents to disk ────────────────────────────────────
for filename, content in documents.items():
    filepath = os.path.join(docs_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content.strip())
    print(f"  ✓ {filename}")

print()
print(f"✓ Created {len(documents)} documents in:")
print(f"  {docs_dir}")
print()
print("✓ SETUP COMPLETE!")
print("  Next step → run:  python test_agent.py")
print("=" * 55)
