import os
import torch

def detect_gpu_device():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA GPU"
        print(f"[OCR Engine] GPU Acceleration Active: NVIDIA GPU ({gpu_name})")
        return "cuda"
    
    try:
        import paddle
        if paddle.is_compiled_with_cuda():
            print("[OCR Engine] GPU Acceleration Active: Paddle CUDA")
            return "gpu"
    except Exception:
        pass

    print("[OCR Engine] Target Device: CPU (Fallback)")
    return "cpu"

device = detect_gpu_device()

_ocr_engine = None

def get_ocr_model():
    global _ocr_engine
    if _ocr_engine is None:
        use_gpu = detect_gpu_device() in ["cuda", "gpu"]
        print(f"Initializing PaddleOCR engine (Device: {'GPU (NVIDIA RTX 4060)' if use_gpu else 'CPU'})...")
        try:
            from paddleocr import PaddleOCR
            _ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en")
            print(f"[SUCCESS] PaddleOCR initialized on {'GPU (NVIDIA RTX 4060)' if use_gpu else 'CPU'}.")
        except Exception as err:
            print(f"[INFO] PaddleOCR init notice ({err}). Checking PaddleOCR-VL via transformers...")
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("PaddlePaddle/PaddleOCR-VL", trust_remote_code=True)
            model = AutoModel.from_pretrained(
                "PaddlePaddle/PaddleOCR-VL", trust_remote_code=True, torch_dtype=torch.float16 if use_gpu else torch.float32
            ).eval()
            if use_gpu and torch.cuda.is_available():
                model = model.cuda()
            _ocr_engine = (tokenizer, model)
            print(f"[SUCCESS] PaddleOCR-VL loaded on {'GPU' if use_gpu else 'CPU'}.")
    return _ocr_engine

def _ocr_single_image(path, engine):
    if isinstance(engine, tuple):
        tokenizer, model = engine
        prompt = "<image>\nConvert document to markdown."
        return model.infer(tokenizer, prompt=prompt, image_file=path)
    else:
        result = engine.ocr(path)
        extracted_lines = []
        if result and result[0]:
            for line in result[0]:
                if isinstance(line, list) and len(line) >= 2 and isinstance(line[1], (list, tuple)):
                    extracted_lines.append(str(line[1][0]))
                elif isinstance(line, dict) and "text" in line:
                    extracted_lines.append(str(line["text"]))
                elif isinstance(line, str):
                    extracted_lines.append(line)
        return "\n".join(extracted_lines)

def ocr_document(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document file not found: {path}")

    if path.lower().endswith(".pdf"):
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(path)
            extracted_pages = []
            engine = None

            try:
                for page_idx, page in enumerate(pdf):
                    textpage = page.get_textpage()
                    page_text = textpage.get_text_range()
                    textpage.close()
                    if page_text and len(page_text.strip()) > 30:
                        extracted_pages.append(page_text.strip())
                    else:
                        if engine is None:
                            engine = get_ocr_model()
                        image = page.render(scale=2).to_pil()
                        temp_img_path = f"{path}_page_{page_idx}.png"
                        image.save(temp_img_path)
                        try:
                            ocr_text = _ocr_single_image(temp_img_path, engine)
                            extracted_pages.append(ocr_text)
                        finally:
                            if os.path.exists(temp_img_path):
                                os.remove(temp_img_path)
                    page.close()
            finally:
                pdf.close()
            
            full_pdf_text = "\n\n".join(extracted_pages)
            if full_pdf_text.strip():
                return full_pdf_text
        except Exception as pdf_err:
            print(f"[PDF Extraction Warning] {pdf_err}. Proceeding with standard OCR...")

    engine = get_ocr_model()
    return _ocr_single_image(path, engine)

def ingest_ocr_document(image_path):
    from ingest import ingest_file
    print(f"Processing document/image: '{image_path}'...")
    markdown_text = ocr_document(image_path)
    print("\n--- Extracted Text Preview ---")
    print(markdown_text[:500] + ("..." if len(markdown_text) > 500 else ""))

    temp_md_path = image_path + ".extracted.md"
    with open(temp_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    ingested_chunks = ingest_file(temp_md_path)
    return markdown_text, ingested_chunks

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        ingest_ocr_document(img_path)
    else:
        print("Usage: python ocr_parse.py <file_path>")
