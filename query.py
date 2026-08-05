import os
import sys
import ollama
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_KEY", "")

def retrieve(query, k=4):
    if not supabase_url or "your-project-id" in supabase_url:
        print("[!] SUPABASE_URL not configured in .env.")
        return []
    supabase = create_client(supabase_url, supabase_key)
    query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
    result = supabase.rpc("match_documents", {"query_embedding": query_embedding, "match_count": k}).execute()
    return result.data

def answer(query, context_override=None):
    if context_override:
        context = context_override
    else:
        chunks = retrieve(query)
        context = "\n\n".join(c["content"] for c in chunks) if chunks else "No relevant context found in database."

    prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {query}
Answer:"""
    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        print(f"Question: {user_query}\n")
        print("Answer:\n" + answer(user_query))
    else:
        test_q = "What are the key components of a RAG pipeline?"
        print(f"Default Test Question: {test_q}\n")
        fallback_ctx = """Key components of a RAG pipeline:
1. Document Ingestion: Converting source documents into chunks.
2. Embedding Generation: Transforming text into vector representations.
3. Vector Database Storage: Storing chunks in pgvector or Supabase.
4. Similarity Retrieval: Fetching top matching chunks.
5. Prompt Augmentation and Synthesis: Generating final answer via LLM."""
        
        if supabase_url and "your-project-id" not in supabase_url:
            print("--- Answering via Supabase Vector Retrieval ---")
            print(answer(test_q))
        else:
            print("--- Answering via Local Sample Context (Supabase URL pending) ---")
            print(answer(test_q, context_override=fallback_ctx))
