# CatVTON FastAPI Backend

Production-ready FastAPI backend for CatVTON virtual try-on service.

## Features

- ✅ **Singleton Model Loading**: Models loaded once at startup, never reloaded per request
- ✅ **GPU Memory Safety**: Single inference at a time with threading lock
- ✅ **CUDA OOM Handling**: Graceful error handling for out-of-memory errors
- ✅ **Automatic Mask Generation**: Uses AutoMasker for clothing type detection
- ✅ **Image Resizing**: Automatic resizing to prevent OOM on smaller GPUs
- ✅ **Health Check Endpoint**: Monitor backend status
- ✅ **CORS Enabled**: Ready for mobile app integration

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure models are downloaded**:
   The first run will download models from HuggingFace if not already cached.

## Running the Server

### Basic Usage

```bash
python app_fastapi.py
```

This starts the server on `http://0.0.0.0:8000` (accessible from all network interfaces).

### Custom Configuration

```bash
python app_fastapi.py \
  --host 0.0.0.0 \
  --port 8000 \
  --base-model "booksforcharlie/stable-diffusion-inpainting" \
  --resume-path "zhengchong/CatVTON" \
  --width 768 \
  --height 1024 \
  --mixed-precision fp16
```

### Command Line Arguments

- `--host`: Host to bind to (default: `0.0.0.0`)
- `--port`: Port to bind to (default: `8000`)
- `--base-model`: Base model path (default: `booksforcharlie/stable-diffusion-inpainting`)
- `--resume-path`: CatVTON checkpoint path (default: `zhengchong/CatVTON`)
- `--width`: Image width (default: `768`)
- `--height`: Image height (default: `1024`)
- `--mixed-precision`: Precision mode - `no`, `fp16`, or `bf16` (default: `fp16`)

## API Endpoints

### Health Check

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "message": "Service is ready"
}
```

### Virtual Try-On

```http
POST /api/try-on
Content-Type: multipart/form-data
```

**Form Data**:
- `person_image` (file): Person photo (JPEG/PNG)
- `cloth_image` (file): Clothing item image (JPEG/PNG)
- `cloth_type` (optional, string): `"upper"`, `"lower"`, or `"overall"` (default: `"upper"`)
- `num_inference_steps` (optional, int): Number of diffusion steps (default: `50`)
- `guidance_scale` (optional, float): CFG strength (default: `2.5`)
- `seed` (optional, int): Random seed (default: `42`, use `-1` for random)

**Response**:
```json
{
  "success": true,
  "imageBase64": "base64_encoded_image_string",
  "message": "Try-on completed successfully"
}
```

**Error Response**:
```json
{
  "detail": "Error message here"
}
```

## GPU Memory Management

### How It Works

1. **Model Loading**: Models are loaded once at application startup in a singleton pattern
2. **Inference Lock**: A threading lock ensures only one inference runs at a time
3. **Memory Cleanup**: GPU cache is cleared after each inference
4. **OOM Protection**: Images are automatically resized if too large
5. **Error Handling**: CUDA OOM errors return proper HTTP 503 responses

### Memory Limits

For **RTX 4050 (6GB VRAM)**:
- Default resolution: `768x1024` (safe)
- Maximum resolution: `768x1024` (recommended)
- Batch size: Always `1` (enforced)

### Preventing OOM

The backend automatically:
- Resizes input images if they exceed `768x1024`
- Clears GPU cache after each inference
- Uses `fp16` precision to reduce memory usage
- Enforces single inference with a lock

## Architecture

### Model Loading Flow

```
Application Startup
    ↓
load_models() called
    ↓
Download models (if needed)
    ↓
Initialize CatVTONPipeline
    ↓
Initialize AutoMasker
    ↓
Models ready for inference
```

### Request Flow

```
POST /api/try-on
    ↓
Check models loaded
    ↓
Read and validate images
    ↓
Resize images if needed
    ↓
Generate mask (AutoMasker)
    ↓
Acquire GPU lock
    ↓
Run inference (with torch.no_grad())
    ↓
Release GPU lock
    ↓
Clear GPU cache
    ↓
Return base64 image
```

## Error Handling

### GPU Busy (503)

When another request is being processed:
```json
{
  "detail": "GPU is busy processing another request. Please try again later."
}
```

### GPU Out of Memory (503)

When image is too large:
```json
{
  "detail": "GPU out of memory. Image may be too large. Error: ..."
}
```

### Models Loading (503)

When models are still loading:
```json
{
  "detail": "Models are still loading. Please wait and try again."
}
```

## Performance

### Expected Performance (RTX 4050, fp16)

- **Model Loading**: ~30-60 seconds (first time)
- **Inference Time**: ~10-20 seconds per request
- **Concurrent Requests**: 1 (enforced by lock)

### Optimization Tips

1. Use `fp16` precision (default)
2. Keep image resolution at `768x1024` or lower
3. Use `num_inference_steps=50` (good balance)
4. Don't call `torch.cuda.empty_cache()` manually (handled automatically)

## Development

### Testing with curl

```bash
curl -X POST "http://localhost:8000/api/try-on" \
  -F "person_image=@person.jpg" \
  -F "cloth_image=@cloth.jpg" \
  -F "cloth_type=upper"
```

### Testing with Python

```python
import requests

url = "http://localhost:8000/api/try-on"
files = {
    "person_image": open("person.jpg", "rb"),
    "cloth_image": open("cloth.jpg", "rb"),
}
data = {
    "cloth_type": "upper",
    "num_inference_steps": 50,
    "guidance_scale": 2.5,
}

response = requests.post(url, files=files, data=data)
result = response.json()
print(result["success"])
```

## Troubleshooting

### Models Not Loading

- Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure sufficient disk space for model downloads
- Check internet connection for HuggingFace downloads

### GPU OOM Errors

- Reduce image resolution in code (modify `width`/`height` defaults)
- Use `fp16` precision (already default)
- Ensure no other processes are using GPU

### Port Already in Use

```bash
# Find process using port 8000
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # macOS/Linux

# Kill process or use different port
python app_fastapi.py --port 8001
```

## Production Deployment

### Using uvicorn directly

```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --workers 1
```

**Note**: Use `--workers 1` to ensure single GPU access. Multiple workers would require GPU sharing.

### Using systemd (Linux)

Create `/etc/systemd/system/catvton.service`:

```ini
[Unit]
Description=CatVTON FastAPI Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/CatVTON
ExecStart=/usr/bin/python3 app_fastapi.py --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable catvton
sudo systemctl start catvton
```

## Security Notes

- CORS is currently set to allow all origins (`allow_origins=["*"]`)
- For production, specify exact origins
- Consider adding authentication/rate limiting
- Use HTTPS in production

## License

See LICENSE file for details.

