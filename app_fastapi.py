"""
FastAPI backend for CatVTON virtual try-on service.
Models are loaded once at startup and reused for all requests.
GPU memory is managed with a lock to ensure single inference at a time.
"""
import os
import io
import base64
import threading
import argparse
from typing import Optional
from contextlib import contextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

# Global model instances (singleton pattern)
_pipeline: Optional[CatVTONPipeline] = None
_automasker: Optional[AutoMasker] = None
_mask_processor: Optional[VaeImageProcessor] = None
_model_lock = threading.Lock()  # Lock for GPU inference (single inference at a time)
_model_loaded = threading.Event()  # Event to signal when models are loaded

# Default configuration
DEFAULT_CONFIG = {
    "base_model_path": "booksforcharlie/stable-diffusion-inpainting",
    "resume_path": "zhengchong/CatVTON",
    "width": 768,
    "height": 1024,
    "mixed_precision": "fp16",  # Use fp16 for RTX 4050 (6GB VRAM)
    "allow_tf32": True,
    "num_inference_steps": 30,  # Reduced for faster inference on RTX 4050
    "guidance_scale": 2.5,
    "seed": 42,
    "cloth_type": "upper",  # Default cloth type
}

app = FastAPI(title="CatVTON API", version="1.0.0")

# CORS middleware for Expo app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_models(config: dict):
    """Load CatVTON models once at startup. This is a singleton pattern."""
    global _pipeline, _automasker, _mask_processor
    
    if _pipeline is not None:
        print("Models already loaded, skipping...")
        return
    
    print("=" * 60)
    print("Loading CatVTON models...")
    print("=" * 60)
    
    try:
        # Download model if needed
        repo_path = config["resume_path"]
        if not os.path.exists(repo_path):
            print(f"Downloading model from HuggingFace: {repo_path}")
            repo_path = snapshot_download(repo_id=repo_path)
            print(f"Model downloaded to: {repo_path}")
        else:
            print(f"Using local model path: {repo_path}")
        
        # Initialize pipeline
        print("Initializing CatVTON pipeline...")
        weight_dtype = init_weight_dtype(config["mixed_precision"])
        _pipeline = CatVTONPipeline(
            base_ckpt=config["base_model_path"],
            attn_ckpt=repo_path,
            attn_ckpt_version="mix",
            weight_dtype=weight_dtype,
            use_tf32=config["allow_tf32"],
            device='cuda',
            skip_safety_check=False,  # Keep safety checker enabled
        )
        print("Pipeline loaded successfully!")
        
        # Initialize AutoMasker
        print("Initializing AutoMasker...")
        _mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True
        )
        _automasker = AutoMasker(
            densepose_ckpt=os.path.join(repo_path, "DensePose"),
            schp_ckpt=os.path.join(repo_path, "SCHP"),
            device='cuda',
        )
        print("AutoMasker loaded successfully!")
        
        # Signal that models are loaded
        _model_loaded.set()
        
        print("=" * 60)
        print("All models loaded successfully!")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("=" * 60)
        
    except Exception as e:
        print(f"ERROR: Failed to load models: {e}")
        raise


@contextmanager
def gpu_inference_lock():
    """
    Context manager to ensure only one inference runs at a time.
    This prevents GPU OOM errors.
    """
    acquired = _model_lock.acquire(timeout=300)  # 5 minute timeout
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GPU is busy processing another request. Please try again later."
        )
    
    try:
        yield
    finally:
        _model_lock.release()


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def resize_if_needed(image: Image.Image, max_size: tuple = (768, 1024)) -> Image.Image:
    """
    Resize image if it's too large to prevent OOM.
    RTX 4050 (6GB) can handle 768x1024, but larger images may cause OOM.
    """
    w, h = image.size
    max_w, max_h = max_size
    
    if w > max_w or h > max_h:
        # Maintain aspect ratio
        ratio = min(max_w / w, max_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        print(f"Resizing image from {w}x{h} to {new_w}x{new_h} to prevent OOM")
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    return image


@app.on_event("startup")
async def startup_event():
    """Load models when FastAPI starts."""
    load_models(DEFAULT_CONFIG)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not _model_loaded.is_set():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "loading", "message": "Models are still loading"}
        )
    return {"status": "healthy", "message": "Service is ready"}


@app.post("/api/try-on")
async def try_on(
    person_image: UploadFile = File(..., description="Person image (JPEG/PNG)"),
    cloth_image: UploadFile = File(..., description="Cloth image (JPEG/PNG)"),
    cloth_type: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
):
    """
    Virtual try-on endpoint.
    
    Args:
        person_image: Person photo (JPEG/PNG)
        cloth_image: Clothing item image (JPEG/PNG)
        cloth_type: Type of clothing - "upper", "lower", or "overall" (default: "upper")
        num_inference_steps: Number of diffusion steps (default: 50)
        guidance_scale: CFG strength (default: 2.5)
        seed: Random seed for reproducibility (default: 42, use -1 for random)
    
    Returns:
        JSON with base64-encoded result image
    """
    # Check if models are loaded
    if not _model_loaded.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are still loading. Please wait and try again."
        )
    
    # Use defaults if not provided
    cloth_type = cloth_type or DEFAULT_CONFIG["cloth_type"]
    # Clamp num_inference_steps to prevent OOM (max 30 for production safety)
    requested_steps = num_inference_steps or DEFAULT_CONFIG["num_inference_steps"]
    num_inference_steps = min(requested_steps, 30)  # Hard limit for production
    if requested_steps > 30:
        print(f"WARNING: num_inference_steps clamped from {requested_steps} to 30 for safety")
    guidance_scale = guidance_scale or DEFAULT_CONFIG["guidance_scale"]
    seed = seed if seed is not None else DEFAULT_CONFIG["seed"]
    width = DEFAULT_CONFIG["width"]
    height = DEFAULT_CONFIG["height"]
    
    try:
        # Read and validate images
        print(f"Processing request: cloth_type={cloth_type}, steps={num_inference_steps}, guidance={guidance_scale}")
        
        # Read person image
        person_bytes = await person_image.read()
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        person_img = resize_if_needed(person_img, (width, height))
        
        # Read cloth image
        cloth_bytes = await cloth_image.read()
        cloth_img = Image.open(io.BytesIO(cloth_bytes)).convert("RGB")
        cloth_img = resize_if_needed(cloth_img, (width, height))
        
        # Resize images to model input size
        person_img = resize_and_crop(person_img, (width, height))
        cloth_img = resize_and_padding(cloth_img, (width, height))
        
        # Generate mask using AutoMasker
        print("Generating mask...")
        mask_result = _automasker(person_img, cloth_type)
        mask = mask_result['mask']
        mask = _mask_processor.blur(mask, blur_factor=9)
        
        # Prepare generator for reproducibility
        generator = None
        if seed != -1:
            generator = torch.Generator(device='cuda').manual_seed(seed)
        
        # Run inference with GPU lock and timeout protection
        print("Running inference...")
        inference_timeout = 120  # 120 seconds (2 min) - RTX 4050 needs more time for 30 steps
        
        with gpu_inference_lock():
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            
            def run_inference_sync():
                """Run inference synchronously - will be executed in thread pool."""
                with torch.no_grad():
                    result_images = _pipeline(
                        image=person_img,
                        condition_image=cloth_img,
                        mask=mask,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        height=height,
                        width=width,
                    )
                return result_images[0]
            
            # Run inference with asyncio timeout (cross-platform)
            try:
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result_image = await asyncio.wait_for(
                        loop.run_in_executor(executor, run_inference_sync),
                        timeout=inference_timeout
                    )
            except asyncio.TimeoutError:
                # Timeout occurred - release lock and return error
                torch.cuda.empty_cache()
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Inference exceeded 120 second timeout. The request was cancelled to prevent GPU lock."
                )
            
            except torch.cuda.OutOfMemoryError as e:
                # Clear cache and return error
                torch.cuda.empty_cache()
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"GPU out of memory. Image may be too large. Error: {str(e)}"
                )
            
            # Clear GPU cache after inference (ONLY after, not before)
            torch.cuda.empty_cache()
        
        # Convert to base64
        result_base64 = image_to_base64(result_image)
        
        print("Inference completed successfully!")
        
        return {
            "success": True,
            "imageBase64": result_base64,
            "message": "Try-on completed successfully"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"GPU out of memory: {str(e)}"
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "CatVTON API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "try_on": "/api/try-on"
        },
        "status": "ready" if _model_loaded.is_set() else "loading"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CatVTON FastAPI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--base-model", type=str, default=DEFAULT_CONFIG["base_model_path"])
    parser.add_argument("--resume-path", type=str, default=DEFAULT_CONFIG["resume_path"])
    parser.add_argument("--width", type=int, default=DEFAULT_CONFIG["width"])
    parser.add_argument("--height", type=int, default=DEFAULT_CONFIG["height"])
    parser.add_argument("--mixed-precision", type=str, default=DEFAULT_CONFIG["mixed_precision"], choices=["no", "fp16", "bf16"])
    
    args = parser.parse_args()
    
    # Update config with CLI args
    DEFAULT_CONFIG.update({
        "base_model_path": args.base_model,
        "resume_path": args.resume_path,
        "width": args.width,
        "height": args.height,
        "mixed_precision": args.mixed_precision,
    })
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

