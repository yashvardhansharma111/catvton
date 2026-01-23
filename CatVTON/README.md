# CatVTON Production API

A production-ready FastAPI service for virtual try-on using diffusion models. This system performs high-quality garment try-on with GPU acceleration and can run fully offline after initial model download.

## Features

- FastAPI based REST API for virtual try-on
- GPU accelerated inference using CUDA
- Fully offline capable after first model download
- Production ready with proper error handling and timeouts
- Optimized memory usage for RTX 4050 (6GB VRAM)
- Single inference lock to prevent GPU OOM
- Automatic image resizing for memory safety

## Requirements

- Python 3.9 or higher
- NVIDIA GPU with CUDA support (minimum 6GB VRAM)
- PyTorch with CUDA
- All dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yashvardhansharma111/catvton.git
cd catvton
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API server:
```bash
python app_fastapi.py
```

The server will start on http://0.0.0.0:8000. On first run, models will be automatically downloaded from HuggingFace (requires internet connection). Subsequent runs can work offline as models are cached locally.

## API Endpoints

### Health Check
```
GET /health
```
Returns the service status and confirms models are loaded.

### Virtual Try-On
```
POST /api/try-on
Content-Type: multipart/form-data
```

**Request Parameters:**
- `person_image` (file): Person photo in JPEG or PNG format
- `cloth_image` (file): Clothing item image in JPEG or PNG format
- `cloth_type` (optional): Type of clothing - "upper", "lower", or "overall" (default: "upper")
- `num_inference_steps` (optional): Number of diffusion steps, max 30 (default: 30)
- `guidance_scale` (optional): CFG strength, 0.0-7.5 (default: 2.5)
- `seed` (optional): Random seed for reproducibility (default: 42, use -1 for random)

**Response:**
```json
{
  "success": true,
  "imageBase64": "base64_encoded_image_string",
  "message": "Try-on completed successfully"
}
```

## Configuration

The API can be configured using command line arguments:

```bash
python app_fastapi.py \
  --host 0.0.0.0 \
  --port 8000 \
  --width 768 \
  --height 1024 \
  --mixed-precision fp16
```

Available options:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port number (default: 8000)
- `--width`: Image width in pixels (default: 768)
- `--height`: Image height in pixels (default: 1024)
- `--mixed-precision`: Precision mode - "no", "fp16", or "bf16" (default: fp16)
- `--base-model`: Base model path (default: booksforcharlie/stable-diffusion-inpainting)
- `--resume-path`: CatVTON checkpoint path (default: zhengchong/CatVTON)

## Model Information

The system uses:
- Base model: Stable Diffusion Inpainting
- CatVTON attention weights for try-on conditioning
- AutoMasker for automatic clothing mask generation
- DensePose and SCHP models for pose and segmentation

Models are downloaded from HuggingFace on first run and cached locally. Total model size is approximately 8-10GB. After initial download, the system can run completely offline.

## Performance

- RTX 4050 (6GB VRAM): ~60-120 seconds per inference
- Higher end GPUs (A100, RTX 3090): ~10-20 seconds per inference
- Memory usage: Optimized for 6GB VRAM with fp16 precision
- Inference timeout: 120 seconds maximum

## Architecture

The system uses a singleton pattern to load models once at startup. All inference runs locally on the GPU with proper memory management:
- Single inference lock prevents concurrent GPU usage
- Automatic cache clearing after each inference
- Image resizing to prevent out-of-memory errors
- Graceful error handling for GPU OOM and timeouts

## Deployment

The API is designed for deployment on:
- Local development machines
- Cloud GPU instances (AWS, GCP, Azure)
- On-premise servers with NVIDIA GPUs
- Docker containers (with GPU passthrough)

For production deployment, consider:
- Using a reverse proxy (nginx, Caddy)
- Setting up SSL/TLS certificates
- Implementing authentication if needed
- Using process managers (systemd, supervisor)
- Monitoring GPU usage and API metrics

## Troubleshooting

**Models not loading:**
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check disk space for model downloads (need ~10GB free)
- Verify internet connection for first-time download

**GPU out of memory:**
- Reduce image resolution using --width and --height flags
- Ensure no other processes are using the GPU
- Use fp16 precision (already default)

**Port already in use:**
- Change port with --port flag
- Or kill the process using port 8000

## License

See LICENSE file for details.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.
