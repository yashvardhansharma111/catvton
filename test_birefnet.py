import torch
from transformers import AutoModelForImageSegmentation

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

print("\nDownloading/loading BiRefNet...")
try:
    m = AutoModelForImageSegmentation.from_pretrained(
        'ZhengPeng7/BiRefNet',
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    m.to('cuda')
    m.eval()
    print("BiRefNet loaded OK:", type(m))
except Exception as e:
    import traceback
    print("FAILED:", e)
    traceback.print_exc()
