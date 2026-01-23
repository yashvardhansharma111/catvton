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
import time
import hashlib
from typing import Optional, Dict, Tuple
from contextlib import contextmanager
from datetime import datetime, timedelta

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
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

# Request tracking for performance monitoring
_request_stats = {
    "total_requests": 0,
    "active_requests": 0,
    "last_request_time": None,
    "lock_wait_times": [],
    "cache_hits": 0,
    "cache_misses": 0,
}
_stats_lock = threading.Lock()  # Lock for stats updates

# Cache for preprocessed person images and masks
# Structure: {cache_key: {"person_img": Image, "mask": Image, "expires_at": datetime}}
_preprocessing_cache: Dict[str, Dict] = {}
_cache_lock = threading.Lock()  # Lock for cache access
CACHE_TTL_MINUTES = 30  # Cache expires after 30 minutes
MAX_CACHE_SIZE = 100  # Maximum number of cached entries

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

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

# Initialize rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    Tracks lock wait times for performance monitoring.
    """
    lock_wait_start = time.time()
    acquired = _model_lock.acquire(timeout=180)  # 3 minute timeout (reduced from 5 min)
    
    if not acquired:
        wait_time = time.time() - lock_wait_start
        print(f"WARNING: GPU lock timeout after {wait_time:.2f}s - request rejected")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GPU is busy processing another request. Please try again in a few moments."
        )
    
    wait_time = time.time() - lock_wait_start
    if wait_time > 0.1:  # Log if waited more than 100ms
        print(f"INFO: GPU lock acquired after {wait_time:.2f}s wait")
        with _stats_lock:
            _request_stats["lock_wait_times"].append(wait_time)
            # Keep only last 100 wait times
            if len(_request_stats["lock_wait_times"]) > 100:
                _request_stats["lock_wait_times"].pop(0)
    
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


def image_hash(image: Image.Image) -> str:
    """
    Generate a hash for an image based on its pixel data.
    Used as cache key to identify identical person images.
    """
    # Convert image to bytes for hashing
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return hashlib.md5(img_bytes.getvalue()).hexdigest()


def get_cache_key(person_img: Image.Image, cloth_type: str) -> str:
    """
    Generate cache key from person image hash and cloth type.
    Same person image + same cloth type = same cache key.
    """
    img_hash = image_hash(person_img)
    return f"{img_hash}_{cloth_type}"


def get_cached_preprocessing(cache_key: str) -> Optional[Tuple[Image.Image, Image.Image]]:
    """
    Retrieve cached preprocessed person image and mask.
    Returns (person_img, mask) if found and not expired, None otherwise.
    """
    with _cache_lock:
        if cache_key not in _preprocessing_cache:
            return None
        
        cache_entry = _preprocessing_cache[cache_key]
        
        # Check if expired
        if datetime.now() > cache_entry["expires_at"]:
            del _preprocessing_cache[cache_key]
            return None
        
        # Return cached data (create copies to avoid modification)
        person_img = cache_entry["person_img"].copy()
        mask = cache_entry["mask"].copy()
        return (person_img, mask)


def set_cached_preprocessing(cache_key: str, person_img: Image.Image, mask: Image.Image):
    """
    Cache preprocessed person image and mask with TTL.
    Automatically evicts oldest entries if cache is full.
    """
    with _cache_lock:
        # Evict expired entries first
        now = datetime.now()
        expired_keys = [
            key for key, entry in _preprocessing_cache.items()
            if now > entry["expires_at"]
        ]
        for key in expired_keys:
            del _preprocessing_cache[key]
        
        # If cache is still full, evict oldest entry
        if len(_preprocessing_cache) >= MAX_CACHE_SIZE:
            # Find oldest entry (lowest expires_at)
            oldest_key = min(
                _preprocessing_cache.keys(),
                key=lambda k: _preprocessing_cache[k]["expires_at"]
            )
            del _preprocessing_cache[oldest_key]
            print(f"Cache evicted oldest entry: {oldest_key[:16]}...")
        
        # Add new entry
        _preprocessing_cache[cache_key] = {
            "person_img": person_img.copy(),
            "mask": mask.copy(),
            "expires_at": datetime.now() + timedelta(minutes=CACHE_TTL_MINUTES)
        }
        print(f"Cache stored entry: {cache_key[:16]}... (cache size: {len(_preprocessing_cache)})")


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
    
    # Check GPU lock status
    lock_available = not _model_lock.locked()
    with _stats_lock:
        avg_wait_time = (
            sum(_request_stats["lock_wait_times"]) / len(_request_stats["lock_wait_times"])
            if _request_stats["lock_wait_times"] else 0
        )
        cache_hits = _request_stats.get("cache_hits", 0)
        cache_misses = _request_stats.get("cache_misses", 0)
        cache_hit_rate = (
            cache_hits / (cache_hits + cache_misses) * 100
            if (cache_hits + cache_misses) > 0 else 0
        )
    
    with _cache_lock:
        cache_size = len(_preprocessing_cache)
    
    return {
        "status": "healthy",
        "message": "Service is ready",
        "gpu_available": lock_available,
        "total_requests": _request_stats["total_requests"],
        "avg_lock_wait_seconds": round(avg_wait_time, 2) if avg_wait_time > 0 else 0,
        "cache_size": cache_size,
        "cache_hit_rate": round(cache_hit_rate, 1)
    }


@app.post("/api/try-on")
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute per IP
async def try_on(
    request: Request,
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
        person_image: Person photo (JPEG/PNG, max 10MB)
        cloth_image: Clothing item image (JPEG/PNG, max 10MB)
        cloth_type: Type of clothing - "upper", "lower", or "overall" (default: "upper")
        num_inference_steps: Number of diffusion steps (default: 30, max: 30)
        guidance_scale: CFG strength (default: 2.5)
        seed: Random seed for reproducibility (default: 42, use -1 for random)
    
    Returns:
        JSON with base64-encoded result image
    """
    request_start_time = time.time()
    
    # Update request stats
    with _stats_lock:
        _request_stats["total_requests"] += 1
        _request_stats["active_requests"] += 1
        _request_stats["last_request_time"] = datetime.now().isoformat()
    
    try:
        # Check if models are loaded
        if not _model_loaded.is_set():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models are still loading. Please wait and try again."
            )
        
        # Security: Validate file sizes (max 10MB per file)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        person_image.file.seek(0, 2)  # Seek to end
        person_size = person_image.file.tell()
        person_image.file.seek(0)  # Reset to start
        
        cloth_image.file.seek(0, 2)  # Seek to end
        cloth_size = cloth_image.file.tell()
        cloth_image.file.seek(0)  # Reset to start
        
        if person_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Person image too large ({person_size / 1024 / 1024:.2f}MB). Maximum size is 10MB."
            )
        
        if cloth_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Cloth image too large ({cloth_size / 1024 / 1024:.2f}MB). Maximum size is 10MB."
            )
        
        # Security: Validate file types
        if person_image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid person image type: {person_image.content_type}. Only JPEG/PNG allowed."
            )
        
        if cloth_image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid cloth image type: {cloth_image.content_type}. Only JPEG/PNG allowed."
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
        
        # Check cache for preprocessed person image and mask
        cache_key = get_cache_key(person_img, cloth_type)
        cached_data = get_cached_preprocessing(cache_key)
        
        if cached_data is not None:
            # Cache hit - reuse preprocessed image and mask
            person_img, mask = cached_data
            print(f"Cache HIT: Reusing preprocessed image and mask for {cache_key[:16]}...")
            with _stats_lock:
                _request_stats["cache_hits"] += 1
        else:
            # Cache miss - generate mask using AutoMasker
            print("Cache MISS: Generating mask...")
            mask_start_time = time.time()
            mask_result = _automasker(person_img, cloth_type)
            mask = mask_result['mask']
            mask = _mask_processor.blur(mask, blur_factor=9)
            mask_time = time.time() - mask_start_time
            print(f"Mask generation took {mask_time:.2f}s")
            
            # Cache the preprocessed image and mask
            set_cached_preprocessing(cache_key, person_img, mask)
            with _stats_lock:
                _request_stats["cache_misses"] += 1
        
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
        
        total_time = time.time() - request_start_time
        print(f"Inference completed successfully! Total time: {total_time:.2f}s")
        
        return {
            "success": True,
            "imageBase64": result_base64,
            "message": "Try-on completed successfully",
            "processing_time_seconds": round(total_time, 2)
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
    finally:
        # Update stats (always decrement active requests)
        with _stats_lock:
            _request_stats["active_requests"] = max(0, _request_stats["active_requests"] - 1)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    with _stats_lock:
        stats = _request_stats.copy()
    
    return {
        "service": "CatVTON API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "try_on": "/api/try-on",
            "stats": "/stats",
            "clear_cache": "/cache (DELETE)"
        },
        "status": "ready" if _model_loaded.is_set() else "loading",
        "rate_limit": "10 requests per minute per IP",
        "max_file_size": "10MB per image"
    }


@app.get("/stats")
async def get_stats():
    """Get API statistics (for monitoring)."""
    with _stats_lock:
        avg_wait = (
            sum(_request_stats["lock_wait_times"]) / len(_request_stats["lock_wait_times"])
            if _request_stats["lock_wait_times"] else 0
        )
        max_wait = max(_request_stats["lock_wait_times"]) if _request_stats["lock_wait_times"] else 0
        cache_hits = _request_stats.get("cache_hits", 0)
        cache_misses = _request_stats.get("cache_misses", 0)
        total_cache_requests = cache_hits + cache_misses
        cache_hit_rate = (
            cache_hits / total_cache_requests * 100
            if total_cache_requests > 0 else 0
        )
    
    with _cache_lock:
        cache_size = len(_preprocessing_cache)
    
    return {
        "total_requests": _request_stats["total_requests"],
        "active_requests": _request_stats["active_requests"],
        "gpu_lock_available": not _model_lock.locked(),
        "last_request_time": _request_stats["last_request_time"],
        "average_lock_wait_seconds": round(avg_wait, 3),
        "max_lock_wait_seconds": round(max_wait, 3),
        "models_loaded": _model_loaded.is_set(),
        "cache": {
            "size": cache_size,
            "max_size": MAX_CACHE_SIZE,
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate_percent": round(cache_hit_rate, 1),
            "ttl_minutes": CACHE_TTL_MINUTES
        }
    }


@app.delete("/cache")
async def clear_cache():
    """Clear the preprocessing cache."""
    with _cache_lock:
        cleared_count = len(_preprocessing_cache)
        _preprocessing_cache.clear()
    
    return {
        "success": True,
        "message": f"Cache cleared. Removed {cleared_count} entries.",
        "cache_size": 0
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

