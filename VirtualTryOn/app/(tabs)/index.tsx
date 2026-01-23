import React, { useState, useRef, useEffect } from 'react';
import { View, StyleSheet, TouchableOpacity, Text, FlatList, ActivityIndicator, Alert, Image as RNImage, Platform } from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { Image } from 'expo-image';
import * as FileSystem from 'expo-file-system';
import Constants from 'expo-constants';
import { ThemedView } from '@/components/themed-view';
import { ThemedText } from '@/components/themed-text';

// Clothing items - matches actual files in assets/clothes/
const CLOTHING_ITEMS = [
  { id: '1', name: 'Colorful Sweatshirt', image: require('@/assets/clothes/colourfull-sweatshirt.jpg') },
  { id: '2', name: 'Green T-Shirt', image: require('@/assets/clothes/green-tshirt.png') },
  { id: '3', name: 'Oversize Black T-Shirt', image: require('@/assets/clothes/over-size-black-t-shirt.png') },
  { id: '4', name: 'Purple Shirt', image: require('@/assets/clothes/purple-shirt.png') },
  { id: '5', name: 'Suit', image: require('@/assets/clothes/suit.png') },
];

// Get the correct API URL based on platform and environment variables
// Environment variables can be set in app.json under "extra" or via .env file
// For Expo, use Constants.expoConfig?.extra to access env vars
const getApiUrl = () => {
  // First, try to get from environment variable (app.json extra field)
  const envApiUrl = Constants.expoConfig?.extra?.API_URL;
  if (envApiUrl) {
    return envApiUrl;
  }
  
  // Fallback: Use platform-specific defaults
  if (__DEV__) {
    if (Platform.OS === 'android') {
      // Android - use laptop IP for real devices
      // Note: 10.0.2.2 is only for Android emulator, not real devices
      // Default: use hostUri IP or fallback to localhost
      const hostUri = Constants.expoConfig?.hostUri;
      if (hostUri) {
        const ip = hostUri.split(':')[0];
        return `http://${ip}:8000/api/try-on`;
      }
      return 'http://10.0.2.2:8000/api/try-on'; // Android emulator
    } else if (Platform.OS === 'ios') {
      // iOS simulator - localhost works
      return 'http://localhost:8000/api/try-on';
    }
  }
  
  // For physical devices, try to use hostUri IP
  const hostUri = Constants.expoConfig?.hostUri;
  if (hostUri) {
    const ip = hostUri.split(':')[0];
    return `http://${ip}:8000/api/try-on`;
  }
  
  // Final fallback
  return 'http://localhost:8000/api/try-on';
};

const API_URL = getApiUrl();
const API_BASE_URL = API_URL.replace('/api/try-on', '');

// Log API URL on component mount for debugging
console.log('=== API Configuration ===');
console.log('Platform:', Platform.OS);
console.log('API URL:', API_URL);
console.log('Host URI:', Constants.expoConfig?.hostUri);
console.log('========================');

export default function HomeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  
  const [capturedPhoto, setCapturedPhoto] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [facing, setFacing] = useState<CameraType>('front');
  const cameraRef = useRef<CameraView>(null);
  const [countdown, setCountdown] = useState<number | null>(null);
  const countdownIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup countdown interval on unmount - MUST be before any early returns
  useEffect(() => {
    return () => {
      if (countdownIntervalRef.current) {
        clearInterval(countdownIntervalRef.current);
      }
    };
  }, []);

  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <ThemedText style={styles.message}>We need your permission to show the camera</ThemedText>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const startCountdown = () => {
    // Don't start if countdown is already running
    if (countdown !== null) {
      return;
    }

    setCountdown(3);
    
    countdownIntervalRef.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev === null || prev <= 1) {
          // Countdown finished, take picture
          if (countdownIntervalRef.current) {
            clearInterval(countdownIntervalRef.current);
            countdownIntervalRef.current = null;
          }
          setCountdown(null);
          takePicture();
          return null;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const cancelCountdown = () => {
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
    setCountdown(null);
  };

  const takePicture = async () => {
    if (cameraRef.current) {
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.8,
          base64: false,
        });
        if (photo?.uri) {
          setCapturedPhoto(photo.uri);
          setResultImage(null);
        }
      } catch (error) {
        console.error('Error taking picture:', error);
        Alert.alert('Error', 'Failed to capture photo');
      }
    }
  };

  const handleClothingSelect = async (clothingItem: typeof CLOTHING_ITEMS[0]) => {
    if (!capturedPhoto) {
      Alert.alert('Error', 'Please capture a photo first');
      return;
    }

    setLoading(true);
    setResultImage(null);

    try {
      console.log('=== Starting Try-On Request ===');
      console.log('API URL:', API_URL);
      console.log('Platform:', Platform.OS);
      console.log('Person Image URI:', capturedPhoto);
      
      // Create FormData
      const formData = new FormData();
      
      // Add person image (FastAPI expects 'person_image')
      const personImageData = {
        uri: capturedPhoto,
        type: 'image/jpeg',
        name: 'person.jpg',
      } as any;
      formData.append('person_image', personImageData);
      console.log('Person image added to FormData');

      // Add cloth image (FastAPI expects 'cloth_image') - handle Metro bundler URL by downloading it first
      const clothSource = RNImage.resolveAssetSource(clothingItem.image);
      let clothUri = clothSource.uri;
      console.log('Cloth Image Source URI:', clothUri);
      
      // If it's a Metro bundler URL (http/https), download it to a local file
      if (clothUri.startsWith('http://') || clothUri.startsWith('https://')) {
        console.log('Downloading cloth image from Metro bundler...');
        // Use documentDirectory as fallback if cacheDirectory is not available
        const cacheDir = (FileSystem as any).cacheDirectory || (FileSystem as any).documentDirectory || '';
        // Extract file extension from the original source, not the Metro URL
        const originalSource = RNImage.resolveAssetSource(clothingItem.image);
        const originalUri = originalSource.uri;
        // Get extension from original path or default to jpg
        let fileExtension = 'jpg';
        if (originalUri.includes('.png')) fileExtension = 'png';
        else if (originalUri.includes('.jpg') || originalUri.includes('.jpeg')) fileExtension = 'jpg';
        
        const fileUri = `${cacheDir}cloth-${Date.now()}.${fileExtension}`;
        console.log('Downloading to:', fileUri);
        const downloadResult = await FileSystem.downloadAsync(clothUri, fileUri);
        // Use the destination URI, not the source URI (which may have query params)
        clothUri = fileUri;
        console.log('Cloth image downloaded to:', clothUri);
      }
      
      // Determine MIME type from file extension
      const getMimeType = (uri: string) => {
        const ext = uri.split('.').pop()?.toLowerCase();
        if (ext === 'png') return 'image/png';
        if (ext === 'jpg' || ext === 'jpeg') return 'image/jpeg';
        return 'image/jpeg';
      };
      
      const clothImageData = {
        uri: clothUri,
        type: getMimeType(clothUri),
        name: `cloth.${clothUri.split('.').pop()?.split('?')[0] || 'jpg'}`,
      } as any;
      
      formData.append('cloth_image', clothImageData);
      console.log('Cloth image added to FormData with URI:', clothUri);

      console.log('Sending POST request to:', API_URL);
      console.log('Request method: POST');
      console.log('Content-Type: multipart/form-data');

      // Test connectivity first with health check
      const healthCheckUrl = `${API_BASE_URL}/health`;
      console.log('Testing backend connectivity to:', healthCheckUrl);
      try {
        const healthResponse = await fetch(healthCheckUrl, {
          method: 'GET',
        });
        console.log('Health check status:', healthResponse.status);
        if (!healthResponse.ok) {
          throw new Error(`Backend health check failed: ${healthResponse.status}`);
        }
        const healthData = await healthResponse.json();
        console.log('Health check response:', healthData);
        if (healthData.status === 'loading') {
          throw new Error('Backend models are still loading. Please wait and try again.');
        }
        console.log('Backend is reachable!');
      } catch (healthError: any) {
        console.error('Health check failed:', healthError);
        const errorMsg = healthError.message || 'Unknown error';
        Alert.alert(
          'Connection Error',
          `Cannot reach backend server.\n\nURL: ${healthCheckUrl}\n\nPlease check:\n1. Backend is running\n2. Internet connection is active\n3. ngrok tunnel is active\n\nError: ${errorMsg}`
        );
        throw new Error(`Backend unreachable: ${errorMsg}`);
      }

      // Send request to backend
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('Response received');
      console.log('Response status:', response.status);
      console.log('Response statusText:', response.statusText);
      console.log('Response ok:', response.ok);
      console.log('Response headers:', JSON.stringify(Object.fromEntries(response.headers.entries())));

      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
          console.log('Error response data:', errorData);
        } catch (e) {
          const text = await response.text();
          console.log('Error response text:', text);
          errorData = { detail: text || 'Unknown error' };
        }
        
        // Handle specific error cases
        const errorMessage = errorData.detail || errorData.error || `HTTP ${response.status}: ${response.statusText}`;
        
        if (response.status === 504 || errorMessage.includes('timeout')) {
          Alert.alert(
            'Processing Timeout',
            'The inference is taking longer than expected. This may happen on slower GPUs. Please try again or use fewer inference steps.'
          );
        } else if (response.status === 503) {
          if (errorMessage.includes('GPU') || errorMessage.includes('busy')) {
            Alert.alert(
              'GPU Busy',
              'The GPU is currently processing another request. Please wait a moment and try again.'
            );
          } else if (errorMessage.includes('loading')) {
            Alert.alert(
              'Backend Loading',
              'The backend models are still loading. Please wait and try again in a few moments.'
            );
          } else {
            Alert.alert('Service Unavailable', errorMessage);
          }
        } else if (response.status === 507 || errorMessage.includes('out of memory')) {
          Alert.alert(
            'GPU Out of Memory',
            'The image may be too large for the GPU. Please try with a smaller image or wait a moment.'
          );
        } else {
          Alert.alert('Error', errorMessage);
        }
        
        throw new Error(errorMessage);
      }

      const data = await response.json();
      console.log('Success! Response data keys:', Object.keys(data));
      console.log('Image base64 length:', data.imageBase64?.length || 0);
      setResultImage(`data:image/jpeg;base64,${data.imageBase64}`);
      console.log('=== Try-On Request Completed Successfully ===');
    } catch (error: any) {
      console.error('=== Try-On Error Details ===');
      console.error('Error type:', error?.constructor?.name);
      console.error('Error message:', error?.message);
      console.error('Error stack:', error?.stack);
      console.error('Full error object:', JSON.stringify(error, Object.getOwnPropertyNames(error)));
      console.error('API URL attempted:', API_URL);
      console.error('Platform:', Platform.OS);
      
      // More detailed error message
      let errorMessage = 'Something went wrong';
      if (error?.message) {
        errorMessage = error.message;
      } else if (error?.toString) {
        errorMessage = error.toString();
      }
      
      Alert.alert('Error', errorMessage);
      console.error('=== End Error Details ===');
    } finally {
      setLoading(false);
    }
  };

  const handleRetake = () => {
    setCapturedPhoto(null);
    setResultImage(null);
    cancelCountdown();
  };

  return (
    <ThemedView style={styles.container}>
      {!capturedPhoto ? (
        <View style={styles.cameraContainer}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing={facing}
          >
            {countdown !== null && (
              <View style={styles.countdownOverlay}>
                <Text style={styles.countdownText}>{countdown}</Text>
                <TouchableOpacity
                  style={styles.cancelButton}
                  onPress={cancelCountdown}
                >
                  <Text style={styles.cancelButtonText}>Cancel</Text>
                </TouchableOpacity>
              </View>
            )}
            <View style={styles.buttonContainer}>
              <TouchableOpacity
                style={[styles.captureButton, countdown !== null && styles.captureButtonDisabled]}
                onPress={startCountdown}
                disabled={countdown !== null}
              >
                <View style={styles.captureButtonInner} />
              </TouchableOpacity>
            </View>
          </CameraView>
        </View>
      ) : (
        <View style={styles.contentContainer}>
          {resultImage ? (
            <View style={styles.resultContainer}>
              <Image source={{ uri: resultImage }} style={styles.resultImage} contentFit="contain" />
              <View style={styles.actionButtons}>
                <TouchableOpacity style={styles.actionButton} onPress={handleRetake}>
                  <Text style={styles.actionButtonText}>Retake Photo</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.actionButton, styles.actionButtonSecondary]}
                  onPress={() => setResultImage(null)}
                >
                  <Text style={styles.actionButtonText}>Try Another</Text>
                </TouchableOpacity>
              </View>
            </View>
          ) : (
            <>
              <View style={styles.previewContainer}>
                <Image source={{ uri: capturedPhoto }} style={styles.previewImage} contentFit="contain" />
                {loading && (
                  <View style={styles.loadingOverlay}>
                    <ActivityIndicator size="large" color="#fff" />
                    <Text style={styles.loadingText}>Processing...</Text>
                  </View>
                )}
              </View>
              
              <View style={styles.clothingContainer}>
                <ThemedText style={styles.clothingTitle}>Select Clothing</ThemedText>
                <FlatList
                  data={CLOTHING_ITEMS}
                  horizontal
                  showsHorizontalScrollIndicator={false}
                  keyExtractor={(item) => item.id}
                  renderItem={({ item }) => (
                    <TouchableOpacity
                      style={styles.clothingItem}
                      onPress={() => handleClothingSelect(item)}
                      disabled={loading}
                    >
                      <Image source={item.image} style={styles.clothingImage} contentFit="cover" />
                      <Text style={styles.clothingName}>{item.name}</Text>
                    </TouchableOpacity>
                  )}
                />
                <TouchableOpacity style={styles.retakeButton} onPress={handleRetake}>
                  <Text style={styles.retakeButtonText}>Retake Photo</Text>
                </TouchableOpacity>
              </View>
            </>
          )}
        </View>
      )}
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 20,
    justifyContent: 'center',
    alignItems: 'flex-end',
  },
  captureButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    borderWidth: 4,
    borderColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#fff',
  },
  message: {
    textAlign: 'center',
    paddingBottom: 10,
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  contentContainer: {
    flex: 1,
  },
  previewContainer: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  previewImage: {
    width: '100%',
    height: '100%',
  },
  resultContainer: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  resultImage: {
    width: '100%',
    height: '100%',
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#fff',
    marginTop: 10,
    fontSize: 16,
  },
  clothingContainer: {
    height: 200,
    backgroundColor: '#fff',
    paddingVertical: 15,
  },
  clothingTitle: {
    fontSize: 18,
    fontWeight: '600',
    paddingHorizontal: 20,
    paddingBottom: 10,
  },
  clothingItem: {
    marginLeft: 15,
    alignItems: 'center',
    width: 120,
  },
  clothingImage: {
    width: 100,
    height: 100,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
  },
  clothingName: {
    marginTop: 8,
    fontSize: 12,
    textAlign: 'center',
  },
  retakeButton: {
    marginTop: 15,
    marginHorizontal: 20,
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  retakeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  actionButtons: {
    flexDirection: 'row',
    padding: 20,
    gap: 15,
  },
  actionButton: {
    flex: 1,
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  actionButtonSecondary: {
    backgroundColor: '#34C759',
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  countdownOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  countdownText: {
    fontSize: 120,
    fontWeight: 'bold',
    color: '#fff',
    textShadowColor: 'rgba(0, 0, 0, 0.75)',
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 10,
  },
  cancelButton: {
    marginTop: 30,
    backgroundColor: 'rgba(255, 59, 48, 0.8)',
    paddingHorizontal: 30,
    paddingVertical: 12,
    borderRadius: 25,
  },
  cancelButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  captureButtonDisabled: {
    opacity: 0.5,
  },
});
