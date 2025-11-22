import { BeautyValues } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface FaceMeta {
  faceCount: number;
  faces?: Array<{
    bbox: { x: number; y: number; width: number; height: number };
    landmarks?: Array<{ x: number; y: number }>;
    confidence?: number;
  }>;
}

export interface BeautyResponse {
  image: string; // base64 data URL
  faceMeta?: FaceMeta | null;
}

export interface FaceAnalysisResponse {
  faceMeta: FaceMeta | null;
}

/**
 * Chuyển đổi data URL thành File object
 */
function dataURLtoFile(dataURL: string, filename: string = 'image.jpg'): File {
  const arr = dataURL.split(',');
  const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/jpeg';
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new File([u8arr], filename, { type: mime });
}

/**
 * Chuyển đổi image source (URL hoặc data URL) thành File
 */
async function imageSourceToFile(imageSource: string, filename: string = 'image.jpg'): Promise<File> {
  // Nếu đã là data URL, chuyển trực tiếp
  if (imageSource.startsWith('data:')) {
    return dataURLtoFile(imageSource, filename);
  }
  
  // Nếu là URL, fetch và chuyển thành File
  const response = await fetch(imageSource);
  const blob = await response.blob();
  return new File([blob], filename, { type: blob.type || 'image/jpeg' });
}

/**
 * Phân tích khuôn mặt trong ảnh
 */
export async function analyzeFace(imageSource: string): Promise<FaceAnalysisResponse> {
  try {
    const file = await imageSourceToFile(imageSource);
    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch(`${API_BASE_URL}/api/beauty/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API error: ${response.status} - ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error analyzing face:', error);
    throw error;
  }
}

/**
 * Chuyển đổi BeautyValues từ frontend sang format backend
 */
function convertBeautyValuesToBackendConfig(values: BeautyValues): any {
  return {
    skinValues: {
      smooth: values.skinValues.smooth,
      whiten: values.skinValues.whiten,
      even: values.skinValues.even,
      korean: values.skinValues.korean,
      texture: values.skinValues.texture,
    },
    faceValues: {
      slim: values.faceValues.slim,
      vline: values.faceValues.vline,
      chinShrink: values.faceValues.chinShrink,
      forehead: values.faceValues.forehead,
      jaw: values.faceValues.jaw,
      noseSlim: values.faceValues.noseSlim,
      noseBridge: values.faceValues.noseBridge,
    },
    eyeValues: {
      enlarge: values.eyeValues.enlarge,
      darkCircle: values.eyeValues.darkCircle,
      depth: values.eyeValues.depth,
      eyelid: values.eyeValues.eyelid,
    },
    mouthValues: {
      smile: values.mouthValues.smile,
      volume: values.mouthValues.volume,
      heart: values.mouthValues.heart,
      teethWhiten: values.mouthValues.teethWhiten,
    },
    hairValues: {
      smooth: values.hairValues.smooth,
      volume: values.hairValues.volume,
      shine: values.hairValues.shine,
    },
    skinMode: values.skinMode,
    faceMode: values.faceMode,
    lipstick: values.lipstick,
    hairColor: values.hairColor,
    eyeMakeup: values.eyeMakeup,
    acneMode: values.acneMode,
  };
}

/**
 * Áp dụng các hiệu ứng làm đẹp thông qua backend API
 */
export async function applyBeauty(
  imageSource: string,
  beautyValues: BeautyValues
): Promise<BeautyResponse> {
  try {
    const file = await imageSourceToFile(imageSource);
    const config = convertBeautyValuesToBackendConfig(beautyValues);
    
    const formData = new FormData();
    formData.append('image', file);
    formData.append('beautyConfig', JSON.stringify(config));

    const response = await fetch(`${API_BASE_URL}/api/beauty/apply`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API error: ${response.status} - ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error applying beauty:', error);
    throw error;
  }
}

/**
 * Sáng da nâng cao (chuyên dụng)
 */
export async function brightenSkin(
  imageSource: string,
  whiten: number,
  preserveTexture: boolean = true,
  adaptiveMode: boolean = true
): Promise<BeautyResponse> {
  try {
    const file = await imageSourceToFile(imageSource);
    
    const formData = new FormData();
    formData.append('image', file);
    formData.append('whiten', whiten.toString());
    formData.append('preserveTexture', preserveTexture.toString());
    formData.append('adaptiveMode', adaptiveMode.toString());

    const response = await fetch(`${API_BASE_URL}/api/beauty/brighten-skin`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API error: ${response.status} - ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error brightening skin:', error);
    throw error;
  }
}

/**
 * Kiểm tra backend có sẵn sàng không
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
    });
    return response.ok;
  } catch (error) {
    console.warn('Backend health check failed:', error);
    return false;
  }
}

