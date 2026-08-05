# RAG App Build Plan — Ollama + Supabase pgvector + LangChain + Groq + Unlimited-OCR

Follow phases in order. Each one produces something that actually runs before you add the next layer.

---

## 0. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- A Supabase project (free tier is fine)
- A Groq API key from [console.groq.com](https://console.groq.com)
- VS Code (or your IDE of choice)

Pull the models you'll need locally:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.1
```

---

## 1. Project setup

```powershell
mkdir rag-app
cd rag-app
python -m venv .venv
.venv\Scripts\activate
```

`requirements.txt`:

```
ollama
supabase
python-dotenv
flask
langchain
langchain-community
langchain-ollama
langchain-groq
langchain-text-splitters
transformers
torch
```

```powershell
pip install -r requirements.txt
```

`.env`:

```
SUPABASE_URL=your-project-url
SUPABASE_KEY=your-service-role-or-anon-key
GROQ_API_KEY=your-groq-key
```

---

## 2. Supabase setup

In the Supabase SQL editor, run:

```sql
create extension if not exists vector;

create table documents (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding vector(768)  -- nomic-embed-text outputs 768 dims
);

create or replace function match_documents (
  query_embedding vector(768),
  match_count int default 5
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language sql stable
as $$
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  order by documents.embedding <=> query_embedding
  limit match_count;
$$;
```

---

## Phase 1 — Prove Ollama + Supabase talk to each other (raw)

`embed_test.py`:

```python
import ollama
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

test_sentences = [
    "The cat sat on the mat.",
    "Python is a popular programming language.",
    "Supabase provides a Postgres database with pgvector support.",
    "Ollama runs large language models locally.",
]

for sentence in test_sentences:
    embedding = ollama.embeddings(model="nomic-embed-text", prompt=sentence)["embedding"]
    supabase.table("documents").insert({"content": sentence, "embedding": embedding}).execute()

print(f"Inserted {len(test_sentences)} rows.")

query = "What handles local model inference?"
query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]

result = supabase.rpc("match_documents", {
    "query_embedding": query_embedding,
    "match_count": 2,
}).execute()

for row in result.data:
    print(round(row["similarity"], 3), row["content"])
```

**Checkpoint:** the top result should be the Ollama sentence. If it's not, stop here and debug before moving on — nothing downstream will work if this doesn't.

---

## Phase 2 — Ingest one real text file (raw)

`ingest.py`:

```python
import ollama
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

def chunk_text(text, chunk_size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    for chunk in chunks:
        embedding = ollama.embeddings(model="nomic-embed-text", prompt=chunk)["embedding"]
        supabase.table("documents").insert({
            "content": chunk,
            "metadata": {"source": path},
            "embedding": embedding,
        }).execute()
    print(f"Ingested {len(chunks)} chunks from {path}")

if __name__ == "__main__":
    ingest_file("sample_notes.txt")
```

Use this to tune `chunk_size`/`overlap` before touching OCR — easier to debug on clean input.

---

## Phase 3 — Query loop end-to-end (raw)

`query.py`:

```python
import ollama
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

def retrieve(query, k=4):
    query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
    result = supabase.rpc("match_documents", {"query_embedding": query_embedding, "match_count": k}).execute()
    return result.data

def answer(query):
    chunks = retrieve(query)
    context = "\n\n".join(c["content"] for c in chunks)
    prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {query}
Answer:"""
    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

if __name__ == "__main__":
    while True:
        q = input("\nAsk something (or 'quit'): ")
        if q == "quit":
            break
        print(answer(q))
```

**This is a complete, working RAG app.** Everything after this is polish.

---

## Phase 4 — Add OCR (Unlimited-OCR)

`ocr_parse.py`:

```python
from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "baidu/Unlimited-OCR", trust_remote_code=True, torch_dtype=torch.bfloat16,
).eval().cuda()

def ocr_document(path):
    return model.infer(
        tokenizer,
        prompt="<image>\n<|grounding|>Convert the document to markdown.",
        image_file=path,
    )
```

> Check the model card on Hugging Face at run time — inference API details can shift as the repo updates (it's only a couple months old).

Feed the returned markdown straight into `chunk_text()` / `ingest_file()`'s logic from Phase 2 — no new pipeline needed.

---

## Phase 5 — Migrate to LangChain

`rag_chain.py` — replaces the manual calls in Phases 2-3 with LangChain equivalents:

```python
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from supabase import create_client

load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = SupabaseVectorStore(
    client=supabase, embedding=embeddings,
    table_name="documents", query_name="match_documents",
)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

def ingest(text, source):
    docs = splitter.create_documents([text], metadatas=[{"source": source}])
    vector_store.add_documents(docs)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template(
    "Answer using only the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)

llm = ChatOllama(model="llama3.1")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# chain.invoke("your question")
```

Compare its output against your Phase 3 raw script on the same question — they should match. If they don't, you know the bug is in the migration, not the underlying logic.

---

## Phase 6 — Swap generation to Groq

One-line change to `rag_chain.py`:

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.environ["GROQ_API_KEY"])
```

> Groq's available model list changes — check `console.groq.com/docs/models` for the current one if `openai/gpt-oss-20b` has been swapped out by the time you get here. Embeddings stay on Ollama; Groq doesn't offer an embeddings endpoint.

---

## Phase 7 — Wrap it in an API

`app.py`:

```python
from flask import Flask, request, jsonify
from rag_chain import chain

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json["question"]
    return jsonify({"answer": chain.invoke(question)})

if __name__ == "__main__":
    app.run(port=5000)
```

Hook this up to a React/Vite frontend from here.

---

## Phase 8 — Eval (later)

Once the app works end-to-end, add Ragas or DeepEval to score retrieval and answer quality against a small set of real questions. Not needed to have a working app — treat it as a separate follow-up pass.

---

## Checklist

- [ ] Phase 1 — raw Ollama + Supabase proof works
- [ ] Phase 2 — real file ingested and chunked
- [ ] Phase 3 — full query loop answers correctly
- [ ] Phase 4 — OCR output flows into the same ingest pipeline
- [ ] Phase 5 — LangChain version matches raw version's output
- [ ] Phase 6 — Groq swapped in for generation
- [ ] Phase 7 — Flask API responding
- [ ] Phase 8 — eval harness in place
