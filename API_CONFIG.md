# API Configuration Guide

This guide explains how to configure the backend API URL for the Virtual Try-On app.

## Configuration Methods

### Method 1: Environment Variable (Recommended)

Add the API URL to `app.json` under the `extra` field:

```json
{
  "expo": {
    "extra": {
      "API_URL": "http://YOUR_IP_ADDRESS:8000/api/try-on"
    }
  }
}
```

Replace `YOUR_IP_ADDRESS` with:
- `localhost` for iOS simulator
- `10.0.2.2` for Android emulator
- Your computer's local IP address (e.g., `192.168.1.100`) for physical devices

### Method 2: Automatic Detection (Default)

If `API_URL` is not set in `app.json`, the app will automatically detect the API URL based on the platform:

- **iOS Simulator**: `http://localhost:8000/api/try-on`
- **Android Emulator**: `http://10.0.2.2:8000/api/try-on`
- **Physical Devices**: Uses the Expo dev server IP address

## Finding Your IP Address

### Windows
```powershell
ipconfig
```
Look for "IPv4 Address" under your active network adapter.

### macOS/Linux
```bash
ifconfig | grep "inet "
```
Or:
```bash
ip addr show
```

## Backend Setup

The FastAPI backend should be running on port 8000 by default. To start it:

```bash
cd CatVTON
python app_fastapi.py --host 0.0.0.0 --port 8000
```

## Testing Connection

The app automatically performs a health check before sending try-on requests. If the backend is unreachable, you'll see an error message with troubleshooting steps.

## Troubleshooting

1. **Connection Refused**: Ensure the backend is running and accessible
2. **Wrong IP**: Make sure both devices are on the same Wi-Fi network
3. **Firewall**: Windows Firewall may block port 8000. Add an exception or temporarily disable it for testing
4. **CORS Errors**: The FastAPI backend includes CORS middleware, but if you see CORS errors, check the backend logs

