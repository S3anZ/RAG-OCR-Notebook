import os
import json
import numpy as np
import ollama
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_KEY", "")

LOCAL_CACHE_FILE = os.path.join(os.path.dirname(__file__), "local_vector_store.json")

def load_local_cache():
    if os.path.exists(LOCAL_CACHE_FILE):
        try:
            with open(LOCAL_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("Error loading local vector cache:", e)
    return []

def save_local_cache(cache_data):
    with open(LOCAL_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

def chunk_text(text, chunk_size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0 or start >= len(text):
            break
    return [c for c in chunks if c.strip()]

def ingest_file(path):
    if not os.path.exists(path):
        print(f"Error: File {path} does not exist.")
        return []

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    doc_name = os.path.basename(path).replace(".extracted.md", "")
    print(f"Divided '{doc_name}' into {len(chunks)} chunks.")

    local_cache = load_local_cache()
    # Remove old entries for this doc_name
    local_cache = [item for item in local_cache if item.get("source") != doc_name]

    new_items = []
    for idx, chunk in enumerate(chunks):
        try:
            emb = ollama.embeddings(model="nomic-embed-text", prompt=chunk)["embedding"]
            new_items.append({
                "source": doc_name,
                "chunk_index": idx,
                "content": chunk,
                "embedding": emb
            })
            print(f"  Chunk #{idx+1} ({len(chunk)} chars) -> Vector Dim: {len(emb)}")
        except Exception as e:
            print(f"Error embedding chunk {idx}: {e}")

    local_cache.extend(new_items)
    save_local_cache(local_cache)

    # Attempt Supabase insert if credentials exist
    if supabase_url and "your-project-id" not in supabase_url:
        try:
            supabase = create_client(supabase_url, supabase_key)
            for item in new_items:
                supabase.table("documents").insert({
                    "content": item["content"],
                    "metadata": {"source": item["source"], "chunk_index": item["chunk_index"]},
                    "embedding": item["embedding"],
                }).execute()
            print(f"[SUCCESS] Ingested {len(new_items)} chunks from '{doc_name}' into Supabase.")
        except Exception as db_err:
            print(f"[!] Supabase DB Notice ({db_err}). Chunks stored in local vector store.")

    return chunks

if __name__ == "__main__":
    ingest_file("sample_notes.txt")
