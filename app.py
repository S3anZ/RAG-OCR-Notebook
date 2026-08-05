from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from rag_chain import ask_rag_with_sources, ingest_text_langchain, delete_document_vectors
from ocr_parse import ingest_ocr_document
import os
import werkzeug.utils

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

library_store = []

def sync_library_from_uploads():
    docs = []
    if os.path.exists(UPLOAD_FOLDER):
        for fname in os.listdir(UPLOAD_FOLDER):
            if not fname.endswith(".extracted.md"):
                fpath = os.path.join(UPLOAD_FOLDER, fname)
                if os.path.isfile(fpath):
                    docs.append({
                        "name": fname,
                        "status": "local",
                        "type": fname.split(".")[-1] if "." in fname else "file",
                        "size": f"{round(os.path.getsize(fpath)/1024, 1)} KB"
                    })
    return docs

def fetch_cloud_library():
    cloud_docs = []
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if supabase_url and "your-project-id" not in supabase_url:
        try:
            from supabase import create_client
            supabase = create_client(supabase_url, supabase_key)
            res = supabase.table("documents").select("metadata").execute()
            sources_map = {}
            for row in (res.data or []):
                meta = row.get("metadata") or {}
                src = meta.get("source")
                if src:
                    sources_map[src] = sources_map.get(src, 0) + 1
            for src, count in sources_map.items():
                cloud_docs.append({
                    "name": src,
                    "status": "cloud",
                    "type": src.split(".")[-1] if "." in src else "file",
                    "size": f"{count} vector chunks"
                })
        except Exception as err:
            print("Supabase library fetch notice:", err)
    return cloud_docs

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "RAG-OCR API"})

@app.route("/library", methods=["GET"])
def get_library():
    local_files = sync_library_from_uploads()
    cloud_files = fetch_cloud_library()
    return jsonify({
        "local_documents": local_files,
        "cloud_documents": cloud_files,
        "documents": local_files + cloud_files
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing 'question' field in JSON payload"}), 400

    db_provider = data.get("db_provider", "cloud")
    llm_provider = data.get("llm_provider", "cloud")

    result = ask_rag_with_sources(question, db_provider=db_provider, llm_provider=llm_provider)
    return jsonify({
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "providers_used": result.get("providers_used", {})
    })

@app.route("/ingest", methods=["POST"])
def ingest():
    data = request.get_json() or {}
    text = data.get("text")
    source = data.get("source", "notes.txt")
    if not text:
        return jsonify({"error": "Missing 'text' field in JSON payload"}), 400

    target_path = os.path.join(app.config["UPLOAD_FOLDER"], source)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(text)

    docs = ingest_text_langchain(text, source_name=source)
    return jsonify({"status": "success", "chunks_processed": len(docs), "source": source})

@app.route("/ocr_ingest", methods=["POST"])
def ocr_ingest():
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        filename = werkzeug.utils.secure_filename(file.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(image_path)
    else:
        data = request.get_json() or {}
        image_path = data.get("image_path")
        if not image_path or not os.path.exists(image_path):
            return jsonify({"error": "No file uploaded or valid 'image_path' provided in JSON"}), 400

    doc_name = os.path.basename(image_path)

    try:
        extracted_text, chunks = ingest_ocr_document(image_path)
        return jsonify({
            "status": "success",
            "file": doc_name,
            "chunks_processed": len(chunks),
            "extracted_text_preview": extracted_text[:300]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete_document", methods=["POST", "DELETE", "OPTIONS"])
def delete_document():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json() or {}
    doc_name = data.get("name") or data.get("filename")
    if not doc_name:
        return jsonify({"error": "Missing 'name' field in payload"}), 400

    doc_basename = os.path.basename(doc_name)
    target_path = os.path.join(app.config["UPLOAD_FOLDER"], doc_basename)
    extracted_path = target_path + ".extracted.md"

    try:
        if os.path.exists(target_path):
            os.remove(target_path)
        if os.path.exists(extracted_path):
            os.remove(extracted_path)
    except Exception as file_err:
        print(f"[Delete Notice] Deferred physical file removal ({file_err}). Proceeding to purge vectors...")

    delete_document_vectors(doc_basename)
    return jsonify({"status": "success", "deleted": doc_basename})

if __name__ == "__main__":
    print("[RAG-OCR Server] Running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
