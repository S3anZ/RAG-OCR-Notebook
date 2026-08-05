import sys
import torch

print("=== GPU & CUDA Diagnostics ===")
print(f"PyTorch Version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU Device: {device_name}")
    print(f"Total VRAM: {total_memory:.2f} GB")
else:
    print("CUDA is not detected on current PyTorch binary. Checking PaddlePaddle CUDA...")

try:
    import paddle
    paddle_cuda = paddle.is_compiled_with_cuda()
    print(f"PaddlePaddle CUDA Available: {paddle_cuda}")
except Exception as e:
    print(f"PaddlePaddle Check: {e}")

print("\n=== Testing PaddleOCR / PaddleOCR-VL Light Engine ===")
try:
    from paddleocr import PaddleOCR
    print("1. Initializing PaddleOCR...")
    use_gpu = cuda_available or (('paddle_cuda' in locals()) and paddle_cuda)
    ocr = PaddleOCR(use_textline_orientation=True, lang="en", use_gpu=use_gpu)
    print(f"   [SUCCESS] PaddleOCR initialized (GPU Accelerated: {use_gpu})!")
except Exception as e:
    print(f"   [NOTE/INFO] Diagnostic test result: {e}")
