import os
import json
import numpy as np
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from supabase import create_client

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_KEY", "")
groq_api_key = os.environ.get("GROQ_API_KEY", "")

LOCAL_CACHE_FILE = os.path.join(os.path.dirname(__file__), "local_vector_store.json")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def get_llm(provider="cloud"):
    if provider == "cloud" and groq_api_key and "your-groq" not in groq_api_key:
        from langchain_groq import ChatGroq
        print("[LLM Backend] Selected: Cloud Groq API (llama-3.3-70b-versatile)")
        return ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
    else:
        print("[LLM Backend] Selected: Local Ollama Model ('llama3.1')")
        return ChatOllama(model="llama3.1")

prompt = ChatPromptTemplate.from_template(
    "Answer the user question concisely using ONLY the document context below. Include footnote numbers like [1], [2] corresponding to sources when referencing factual details.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def cosine_similarity(v1, v2):
    vec1 = np.array(v1, dtype=np.float32)
    vec2 = np.array(v2, dtype=np.float32)
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 > 0 and norm2 > 0:
        return float(dot / (norm1 * norm2))
    return 0.0

def local_vector_search(query, k=4):
    if not os.path.exists(LOCAL_CACHE_FILE):
        return []
    try:
        with open(LOCAL_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception as e:
        print("Error reading local vector store:", e)
        return []

    if not cache:
        return []

    query_emb = embeddings.embed_query(query)
    scored_items = []
    for item in cache:
        sim = cosine_similarity(query_emb, item.get("embedding", []))
        scored_items.append({
            "source": item.get("source", "document"),
            "chunk_index": item.get("chunk_index", 0),
            "content": item.get("content", ""),
            "similarity": sim
        })

    scored_items.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_items[:k]

def ask_rag_with_sources(question, db_provider="cloud", llm_provider="cloud"):
    llm = get_llm(provider=llm_provider)
    answer_text = ""
    sources = []

    # 1. Cloud Supabase pgvector search
    if db_provider == "cloud" and supabase_url and "your-project-id" not in supabase_url:
        try:
            print("[Vector Database] Selected: Cloud Supabase pgvector store")
            supabase = create_client(supabase_url, supabase_key)
            vector_store = SupabaseVectorStore(
                client=supabase,
                embedding=embeddings,
                table_name="documents",
                query_name="match_documents",
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            answer_text = chain.invoke(question)

            query_embedding = embeddings.embed_query(question)
            res = supabase.rpc("match_documents", {"query_embedding": query_embedding, "match_count": 4}).execute()
            for idx, item in enumerate(res.data or []):
                meta = item.get("metadata") or {}
                sources.append({
                    "id": idx + 1,
                    "file": meta.get("source", "document"),
                    "chunk": meta.get("chunk_index", 0),
                    "content": item.get("content", ""),
                    "similarity": round(item.get("similarity", 0.0), 3)
                })
        except Exception as err:
            print(f"\n[!] Supabase Cloud Notice: ({err}). Falling back to local vector store search...")

    # 2. Local Vector Store (Scoped strictly to uploads/ folder)
    if not answer_text or not sources or db_provider == "local":
        print("[Vector Database] Selected: Local Vector Store (Scoped to uploads/ directory)")
        local_results = local_vector_search(question, k=4)
        if local_results:
            context_blocks = []
            sources = []
            for idx, item in enumerate(local_results):
                fn_num = idx + 1
                context_blocks.append(f"[{fn_num}] Document: {item['source']}\n{item['content']}")
                sources.append({
                    "id": fn_num,
                    "file": item["source"],
                    "chunk": item["chunk_index"],
                    "content": item["content"],
                    "similarity": round(item["similarity"], 3)
                })

            full_context = "\n\n".join(context_blocks)
            formatted_prompt = prompt.format(context=full_context, question=question)
            res = llm.invoke(formatted_prompt)
            answer_text = res.content if hasattr(res, 'content') else str(res)
        else:
            answer_text = "No uploaded documents found in vector store. Please upload a file to the uploads/ directory."

    return {
        "answer": answer_text,
        "sources": sources,
        "providers_used": {
            "db": "cloud_supabase" if (db_provider == "cloud" and sources) else "local_uploads",
            "llm": "groq_cloud" if (llm_provider == "cloud" and groq_api_key) else "ollama_local"
        }
    }

def ingest_text_langchain(text, source_name="manual_input"):
    from ingest import chunk_text, load_local_cache, save_local_cache
    chunks = chunk_text(text)
    cache = load_local_cache()
    cache = [item for item in cache if item.get("source") != source_name]

    for idx, chunk in enumerate(chunks):
        emb = embeddings.embed_query(chunk)
        cache.append({
            "source": source_name,
            "chunk_index": idx,
            "content": chunk,
            "embedding": emb
        })
    save_local_cache(cache)
    return chunks

def delete_document_vectors(filename):
    if os.path.exists(LOCAL_CACHE_FILE):
        try:
            with open(LOCAL_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            doc_base = os.path.basename(filename).replace(".extracted.md", "")
            filtered = [item for item in cache if item.get("source") != doc_base and item.get("source") != filename]
            with open(LOCAL_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(filtered, f, ensure_ascii=False, indent=2)
            print(f"[SUCCESS] Removed '{filename}' from local vector store.")
        except Exception as e:
            print("Error purging local vector cache:", e)

    if supabase_url and "your-project-id" not in supabase_url:
        try:
            supabase = create_client(supabase_url, supabase_key)
            supabase.table("documents").delete().filter("metadata->>source", "like", f"%{filename}%").execute()
            print(f"[SUCCESS] Deleted vectors for '{filename}' from Supabase table 'documents'.")
        except Exception as err:
            print(f"[Delete Notice] Supabase purge notice ({err}).")

    return True

if __name__ == "__main__":
    res = ask_rag_with_sources("What is RAG?", db_provider="local", llm_provider="cloud")
    print("Answer:\n", res["answer"])
    print("Providers:\n", res["providers_used"])
