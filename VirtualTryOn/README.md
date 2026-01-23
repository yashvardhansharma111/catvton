# VirtualTryOn - Complete Virtual Try-On System

A complete virtual try-on system consisting of a FastAPI backend (CatVTON) and an Expo React Native mobile app. This system allows users to try on clothes virtually using AI-powered diffusion models.

## Repository Structure

This repository contains both the backend and frontend components:

- **Backend (CatVTON)**: FastAPI service for virtual try-on inference
- **Frontend (VirtualTryOn)**: Expo React Native mobile application

## Backend - CatVTON Production API

A production-ready FastAPI service for virtual try-on using diffusion models. This system performs high-quality garment try-on with GPU acceleration and can run fully offline after initial model download.

### Features

- FastAPI based REST API for virtual try-on
- GPU accelerated inference using CUDA
- Fully offline capable after first model download
- Production ready with proper error handling and timeouts
- Rate limiting and security checks
- Intelligent caching for improved performance
- Optimized memory usage for RTX 4050 (6GB VRAM)
- Single inference lock to prevent GPU OOM
- Automatic image resizing for memory safety

### Requirements

- Python 3.9 or higher
- NVIDIA GPU with CUDA support (minimum 6GB VRAM)
- PyTorch with CUDA
- All dependencies listed in requirements.txt

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API server:
```bash
python app_fastapi.py
```

The server will start on http://0.0.0.0:8000. On first run, models will be automatically downloaded from HuggingFace (requires internet connection). Subsequent runs can work offline as models are cached locally.

### API Endpoints

- `GET /health` - Health check endpoint
- `POST /api/try-on` - Virtual try-on endpoint
- `GET /stats` - API statistics and cache metrics
- `DELETE /cache` - Clear preprocessing cache

See `README_FASTAPI.md` for detailed API documentation.

## Frontend - VirtualTryOn Mobile App

An Expo React Native mobile application that provides a user-friendly interface for virtual try-on.

### Features

- Image selection from camera or gallery
- Real-time try-on processing
- Loading and error state management
- API integration with FastAPI backend
- Environment-based API configuration

### Requirements

- Node.js 18 or higher
- npm or yarn
- Expo CLI
- iOS Simulator / Android Emulator or physical device

### Installation

1. Install dependencies:
```bash
npm install
```

2. Configure API URL in `app.json`:
```json
{
  "extra": {
    "API_URL": "https://your-backend-url.com/api/try-on"
  }
}
```

3. Start the app:
```bash
npm start
```

See `README_SETUP.md` and `API_CONFIG.md` for detailed setup instructions.

## Quick Start

### Backend Setup
```bash
cd /path/to/repo
pip install -r requirements.txt
python app_fastapi.py
```

### Frontend Setup
```bash
cd /path/to/repo
npm install
npm start
```

## Architecture

### Backend Architecture
- Models loaded once at startup (singleton pattern)
- Single GPU inference lock prevents concurrent usage
- Intelligent caching for preprocessed images and masks
- Rate limiting (10 requests/minute per IP)
- Automatic cache clearing after inference

### Frontend Architecture
- Expo Router for navigation
- React Native components for UI
- Image picker for camera/gallery access
- Multipart form data for API uploads
- Environment-based configuration

## Performance

- RTX 4050 (6GB VRAM): ~60-120 seconds per inference
- RTX 4090 (24GB VRAM): ~10 seconds per inference
- Cache hit saves 2-3 seconds per request
- Memory usage: Optimized for 6GB VRAM with fp16 precision

## Deployment

### Backend Deployment
- Local development machines
- Cloud GPU instances (AWS, GCP, Azure, RunPod)
- On-premise servers with NVIDIA GPUs
- Docker containers (with GPU passthrough)

### Frontend Deployment
- Expo Go for development
- EAS Build for production apps
- App Store / Google Play Store distribution

## License

See LICENSE file for details.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.
