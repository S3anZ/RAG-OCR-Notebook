import os
import ollama
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_KEY", "")

if not supabase_url or "your-project-id" in supabase_url:
    print("\n[!] ATTENTION: Please update .env with your real SUPABASE_URL and SUPABASE_KEY before executing Supabase calls.\n")

print("--- Phase 1: Raw Ollama + Vector Embedding Test ---")
test_sentences = [
    "The cat sat on the mat.",
    "Python is a popular programming language.",
    "Supabase provides a Postgres database with pgvector support.",
    "Ollama runs large language models locally.",
]

print("1. Testing Ollama embeddings generator with 'nomic-embed-text'...")
sample_embed = ollama.embeddings(model="nomic-embed-text", prompt="Testing vector generation")["embedding"]
print(f"   [SUCCESS] Embedding generated. Dimension size: {len(sample_embed)}")

if supabase_url and "your-project-id" not in supabase_url:
    supabase = create_client(supabase_url, supabase_key)
    print("2. Inserting test sentences into Supabase...")
    for sentence in test_sentences:
        embedding = ollama.embeddings(model="nomic-embed-text", prompt=sentence)["embedding"]
        supabase.table("documents").insert({"content": sentence, "embedding": embedding}).execute()
    print(f"   Inserted {len(test_sentences)} rows.")

    query = "What handles local model inference?"
    print(f"3. Performing vector search query: '{query}'")
    query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]

    result = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_count": 2,
    }).execute()

    print("4. Results from Supabase RPC match_documents:")
    for row in result.data:
        print(f"   Score: {round(row['similarity'], 3)} | Content: {row['content']}")
else:
    print("2. Skipping Supabase DB insertion (SUPABASE_URL not configured yet).")
