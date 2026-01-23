# Virtual Try-On Frontend Setup

React Native Expo app for Virtual Try-On MVP.

## Setup

1. **Install dependencies:**
   ```bash
   cd VirtualTryOn
   npm install
   ```

2. **Add clothing images:**
   
   Place your clothing images (PNG/JPG) in `assets/clothes/` folder.
   
   Update the `CLOTHING_ITEMS` array in `app/(tabs)/index.tsx` to match your image filenames:
   ```typescript
   const CLOTHING_ITEMS = [
     { id: '1', name: 'Shirt 1', image: require('@/assets/clothes/shirt1.jpg') },
     { id: '2', name: 'Shirt 2', image: require('@/assets/clothes/shirt2.jpg') },
     // Add more items...
   ];
   ```

3. **Configure backend URL:**
   
   Update the `API_URL` constant in `app/(tabs)/index.tsx` if your backend runs on a different URL:
   ```typescript
   const API_URL = 'http://localhost:3000/try-on';
   // For physical device, use your computer's IP:
   // const API_URL = 'http://192.168.1.XXX:3000/try-on';
   ```

4. **Start the app:**
   ```bash
   npm start
   ```
   
   Then press:
   - `i` for iOS simulator
   - `a` for Android emulator
   - Scan QR code with Expo Go app on your phone

## App Flow

1. User opens app → Camera screen appears
2. User captures front-facing photo
3. Horizontal scroll list of clothing items appears
4. User taps a clothing item
5. App sends photo + clothing image to backend
6. Loading indicator shows while processing
7. Generated image displays
8. User can try another clothing item or retake photo

## Permissions

The app requires camera permissions. Grant permission when prompted.

## Troubleshooting

- **Camera not working:** Ensure camera permissions are granted in device settings
- **API connection failed:** 
  - Check backend is running
  - For physical device, ensure device and computer are on same network
  - Update API_URL with correct IP address
- **Clothing images not showing:** Ensure images exist in `assets/clothes/` and filenames match the CLOTHING_ITEMS array

