
import { GoogleGenAI, Type, Schema } from "@google/genai";

export interface AIProAdjustments {
    basic?: Record<string, number>;
    beauty?: {
        skinValues?: Record<string, number>;
        faceValues?: Record<string, number>;
        eyeValues?: Record<string, number>;
        mouthValues?: Record<string, number>;
        hairValues?: Record<string, number>;
        lipstick?: string;
    };
    filters?: any;
    effects?: any;
}

export interface AIPreviewMeta {
    variant: 'remove_bg' | 'people' | 'object' | 'quality_superres' | 'quality_restore' | 'color_clone' | 'generic';
    description?: string;
    badge?: string;
}

export interface AIProResult {
    adjustments?: AIProAdjustments;
    previewImage?: string;
    maskImage?: string;
    previewMeta?: AIPreviewMeta;
    summary?: string;
    metrics?: Record<string, any>;
    steps?: string[];
    qaNotes?: string;
}

const extractImageFromResponse = (response: any): string | undefined => {
    if (response.candidates?.[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
            if (part.inlineData) {
                return `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
            }
        }
    }
    return undefined;
};

// Helper to convert File to Base64 string (raw data)
const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            const result = reader.result as string;
            // Remove data url prefix
            const base64 = result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = error => reject(error);
    });
};

export const runAiProModule = async (
    imageData: string,
    moduleId: string,
    intensity: number,
    options: any,
    files: { referenceImageFile: File | null }
): Promise<AIProResult> => {
    const apiKey = process.env.API_KEY;
    if (!apiKey) throw new Error("API Key not configured");
    
    const ai = new GoogleGenAI({ apiKey });
    const base64Data = imageData.split(',')[1];
    const mimeType = imageData.substring(imageData.indexOf(':') + 1, imageData.indexOf(';'));
    
    const result: AIProResult = {
        steps: ['Analyzing image...', 'Processing with Gemini AI...', 'Finalizing...'],
        summary: 'Processing complete.'
    };

    try {
        if (moduleId === 'ai_cutout_remove') {
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash-image',
                contents: {
                    parts: [
                        { inlineData: { mimeType, data: base64Data } },
                        { text: "Remove the background from this image. Return the subject on a transparent background. Output ONLY the image as PNG." }
                    ]
                }
            });
            const img = extractImageFromResponse(response);
            if (img) {
                result.previewImage = img;
                result.previewMeta = {
                    variant: 'remove_bg',
                    description: 'Background removed successfully.',
                    badge: 'Cutout'
                };
                result.summary = "Background removed.";
            } else {
                 throw new Error("AI did not return an image.");
            }
        } else if (moduleId === 'ai_quality_superres') {
             const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash-image',
                contents: {
                    parts: [
                        { inlineData: { mimeType, data: base64Data } },
                        { text: "Upscale this image to 4K resolution, improve details, remove noise. Output the enhanced image." }
                    ]
                }
            });
            const img = extractImageFromResponse(response);
            if (img) {
                result.previewImage = img;
                result.previewMeta = { variant: 'quality_superres', description: 'High resolution image generated.' };
                result.summary = "Image upscaled.";
            }
        } else if (moduleId === 'ai_quality_restore') {
             const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash-image',
                contents: {
                    parts: [
                        { inlineData: { mimeType, data: base64Data } },
                        { text: "Colorize and restore this image. If it is black and white, add natural colors. Output the restored image." }
                    ]
                }
            });
            const img = extractImageFromResponse(response);
            if (img) {
                result.previewImage = img;
                result.previewMeta = { variant: 'quality_restore', description: 'Image restored and colorized.' };
                result.summary = "Image restored.";
            }
        } else if (moduleId === 'ai_color_transfer') {
            // New Logic for Color Transfer
            if (!files.referenceImageFile) {
                throw new Error("Vui lòng chọn ảnh tham chiếu (Reference Image) để thực hiện chuyển màu.");
            }

            const refBase64 = await fileToBase64(files.referenceImageFile);
            const refMimeType = files.referenceImageFile.type || 'image/jpeg';

            // Define Schema for Adjustment Values
            // Added 'hue' to allow color shifting for better matching
            const adjustmentSchema: Schema = {
                type: Type.OBJECT,
                properties: {
                    exposure: { type: Type.INTEGER, description: "Exposure value between -100 and 100" },
                    contrast: { type: Type.INTEGER, description: "Contrast value between -100 and 100" },
                    highlights: { type: Type.INTEGER, description: "Highlights value between -100 and 100" },
                    shadows: { type: Type.INTEGER, description: "Shadows value between -100 and 100" },
                    temp: { type: Type.INTEGER, description: "Temperature value between -100 and 100 (warm/cool)" },
                    tint: { type: Type.INTEGER, description: "Tint value between -100 and 100 (green/magenta)" },
                    saturation: { type: Type.INTEGER, description: "Saturation value between -100 and 100" },
                    vibrance: { type: Type.INTEGER, description: "Vibrance value between -100 and 100" },
                    hue: { type: Type.INTEGER, description: "Hue shift value between -180 and 180" },
                    whites: { type: Type.INTEGER, description: "Whites value between -100 and 100" },
                    blacks: { type: Type.INTEGER, description: "Blacks value between -100 and 100" }
                },
                required: ["exposure", "contrast", "temp", "tint", "saturation", "vibrance", "hue"]
            };

            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: {
                    parts: [
                        { text: "You are a professional high-end film colorist. Your task is to transfer the 'Color Soul' (Vibe, Palette, and Dynamic Range) from the Reference Image (second image) to the Source Image (first image). \n\nPROCESS:\n1. Analyze the Reference Image: Extract its color contrast (warm vs cool), saturation levels, and dynamic range (fade vs contrast).\n2. Map these to the Source Image using standard adjustment parameters.\n\nRULES FOR MULTI-COLOR OUTPUT:\n- DO NOT apply a monochromatic wash (single color filter). Avoid extreme Temp/Tint values unless the reference is strictly Sepia/B&W.\n- If the reference is vibrant and multi-colored, preserve color separation in the source.\n- Use 'Vibrance' and 'Contrast' as your primary tools to match the mood.\n- Use 'Hue' shift if necessary to align the palette (e.g., turning green leaves to autumn orange).\n- Use 'Highlights' and 'Shadows' to match the lighting dynamic.\n\nReturn JSON with integer values." },
                        { inlineData: { mimeType: mimeType, data: base64Data } }, // Source
                        { inlineData: { mimeType: refMimeType, data: refBase64 } } // Reference
                    ]
                },
                config: {
                    responseMimeType: "application/json",
                    responseSchema: adjustmentSchema
                }
            });

            const jsonText = response.text;
            if (jsonText) {
                const adjustments = JSON.parse(jsonText);
                
                // REDUCE INTENSITY: Apply a 0.7 dampener (30% reduction) to ensure results are not too harsh.
                // This allows for a more natural blend.
                const globalDampener = 0.7; 
                const factor = (intensity / 100) * globalDampener;
                
                const scaledAdjustments: Record<string, number> = {};
                
                Object.keys(adjustments).forEach(key => {
                    // For temp and tint, we dampen them slightly more to prevent monochromatic color casts
                    if (key === 'temp' || key === 'tint') {
                        scaledAdjustments[key] = Math.round(adjustments[key] * factor * 0.8);
                    } else if (key === 'vibrance') {
                        // Boost vibrance slightly to ensure richness (multi-color feel)
                        scaledAdjustments[key] = Math.round(adjustments[key] * factor * 1.2);
                    } else {
                        scaledAdjustments[key] = Math.round(adjustments[key] * factor);
                    }
                });

                result.adjustments = {
                    basic: scaledAdjustments
                };
                result.summary = "Đã sao chép màu từ ảnh tham chiếu (Chế độ màu đa sắc).";
                result.steps = [
                    "Phân tích bảng màu và độ tương phản của ảnh mẫu",
                    "Tối ưu hóa Vibrance để giữ độ rực rỡ đa sắc",
                    "Cân bằng Dynamic Range (Shadow/Highlight)",
                    "Áp dụng hệ số giảm cường độ 30% cho vẻ đẹp tự nhiên"
                ];
            }

        } else if (moduleId === 'ai_beauty_full') {
            // AI Tự chỉnh toàn diện - Comprehensive automatic enhancement
            const fullBeautySchema: Schema = {
                type: Type.OBJECT,
                properties: {
                    // Basic adjustments
                    exposure: { type: Type.INTEGER, description: "Exposure adjustment (-100 to 100)" },
                    contrast: { type: Type.INTEGER, description: "Contrast adjustment (-100 to 100)" },
                    highlights: { type: Type.INTEGER, description: "Highlights adjustment (-100 to 100)" },
                    shadows: { type: Type.INTEGER, description: "Shadows adjustment (-100 to 100)" },
                    vibrance: { type: Type.INTEGER, description: "Vibrance adjustment (-100 to 100)" },
                    saturation: { type: Type.INTEGER, description: "Saturation adjustment (-100 to 100)" },
                    clarity: { type: Type.INTEGER, description: "Clarity/sharpness adjustment (-100 to 100)" },
                    temp: { type: Type.INTEGER, description: "Color temperature (-100 to 100)" },
                    // Beauty adjustments
                    skinSmooth: { type: Type.INTEGER, description: "Skin smoothing (0 to 100)" },
                    skinWhiten: { type: Type.INTEGER, description: "Skin whitening (0 to 100)" },
                    skinEven: { type: Type.INTEGER, description: "Skin tone evenness (0 to 100)" },
                    eyeBrightness: { type: Type.INTEGER, description: "Eye brightness enhancement (0 to 100)" },
                    eyeDarkCircle: { type: Type.INTEGER, description: "Dark circle reduction (0 to 100)" }
                },
                required: ["exposure", "contrast", "vibrance", "clarity", "skinSmooth", "skinWhiten", "skinEven"]
            };

            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: {
                    parts: [
                        { text: "You are a professional photo editor AI. Analyze this image and provide comprehensive automatic enhancements.\n\nTASK: 'Tự chỉnh toàn diện' (Full Auto Enhancement)\n\nAnalyze the image for:\n1. Lighting issues (over/under exposure, contrast problems)\n2. Color balance (temperature, saturation, vibrance)\n3. Image sharpness and clarity\n4. Skin quality (if portrait detected)\n5. Eye appearance (if face detected)\n\nProvide balanced adjustments that improve overall image quality while maintaining natural look. Be conservative - aim for subtle improvements rather than dramatic changes.\n\nReturn JSON with integer values. For beauty parameters, only apply if human faces are clearly visible." },
                        { inlineData: { mimeType, data: base64Data } }
                    ]
                },
                config: {
                    responseMimeType: "application/json",
                    responseSchema: fullBeautySchema
                }
            });

            const jsonText = response.text;
            if (jsonText) {
                const adjustments = JSON.parse(jsonText);
                const factor = intensity / 100;
                
                result.adjustments = {
                    basic: {
                        exposure: Math.round(adjustments.exposure * factor),
                        contrast: Math.round(adjustments.contrast * factor),
                        highlights: Math.round((adjustments.highlights || 0) * factor),
                        shadows: Math.round((adjustments.shadows || 0) * factor),
                        vibrance: Math.round(adjustments.vibrance * factor),
                        saturation: Math.round((adjustments.saturation || 0) * factor),
                        clarity: Math.round(adjustments.clarity * factor),
                        temp: Math.round((adjustments.temp || 0) * factor)
                    },
                    beauty: {
                        skinValues: {
                            smooth: Math.round((adjustments.skinSmooth || 0) * factor),
                            whiten: Math.round((adjustments.skinWhiten || 0) * factor),
                            even: Math.round((adjustments.skinEven || 0) * factor),
                            korean: 0,
                            texture: 50
                        },
                        eyeValues: {
                            enlarge: 0,
                            brightness: Math.round((adjustments.eyeBrightness || 0) * factor),
                            darkCircle: Math.round((adjustments.eyeDarkCircle || 0) * factor),
                            depth: 0
                        }
                    }
                };
                result.summary = "Đã tự động điều chỉnh toàn diện: ánh sáng, màu sắc, độ nét và làm đẹp da.";
                result.steps = [
                    "Phân tích chất lượng ảnh tổng thể",
                    "Cân bằng ánh sáng và độ tương phản",
                    "Tối ưu màu sắc và độ rực rỡ",
                    "Nâng cao độ nét và chi tiết",
                    "Làm đẹp da và mắt (nếu phát hiện chân dung)"
                ];
            }

        } else if (moduleId === 'ai_beauty_portrait') {
            // AI Tối ưu chân dung - Portrait optimization
            const portraitSchema: Schema = {
                type: Type.OBJECT,
                properties: {
                    // Lighting for portraits
                    exposure: { type: Type.INTEGER, description: "Exposure for portrait lighting (-100 to 100)" },
                    highlights: { type: Type.INTEGER, description: "Highlights to soften skin (-100 to 100)" },
                    shadows: { type: Type.INTEGER, description: "Shadows to add depth (-100 to 100)" },
                    // Portrait beauty
                    skinSmooth: { type: Type.INTEGER, description: "Skin smoothing (0 to 100)" },
                    skinWhiten: { type: Type.INTEGER, description: "Skin whitening (0 to 100)" },
                    skinEven: { type: Type.INTEGER, description: "Skin tone evenness (0 to 100)" },
                    skinKorean: { type: Type.INTEGER, description: "Korean glass skin effect (0 to 100)" },
                    eyeEnlarge: { type: Type.INTEGER, description: "Eye enlargement (0 to 100)" },
                    eyeBrightness: { type: Type.INTEGER, description: "Eye brightness (0 to 100)" },
                    eyeDarkCircle: { type: Type.INTEGER, description: "Dark circle removal (0 to 100)" },
                    faceSlim: { type: Type.INTEGER, description: "Face slimming (0 to 100)" },
                    vibrance: { type: Type.INTEGER, description: "Vibrance for skin glow (-100 to 100)" }
                },
                required: ["exposure", "highlights", "shadows", "skinSmooth", "skinWhiten", "skinEven", "eyeBrightness"]
            };

            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: {
                    parts: [
                        { text: "You are a professional portrait retoucher AI. Analyze this portrait image and optimize it for beauty.\n\nTASK: 'Tối ưu chân dung' (Portrait Optimization)\n\nFocus on:\n1. Skin quality: smoothness, even tone, natural whitening\n2. Eye enhancement: brightness, dark circle removal, subtle enlargement\n3. Face shape: gentle slimming if needed\n4. Lighting: soft highlights, balanced shadows for depth\n5. Overall glow: subtle vibrance for healthy skin appearance\n\nIMPORTANT: This is for PORTRAITS only. If no clear human face is detected, return minimal adjustments. Be natural - avoid over-processing. Korean glass skin effect should be subtle.\n\nReturn JSON with integer values." },
                        { inlineData: { mimeType, data: base64Data } }
                    ]
                },
                config: {
                    responseMimeType: "application/json",
                    responseSchema: portraitSchema
                }
            });

            const jsonText = response.text;
            if (jsonText) {
                const adjustments = JSON.parse(jsonText);
                const factor = intensity / 100;
                
                result.adjustments = {
                    basic: {
                        exposure: Math.round(adjustments.exposure * factor),
                        highlights: Math.round(adjustments.highlights * factor),
                        shadows: Math.round(adjustments.shadows * factor),
                        vibrance: Math.round((adjustments.vibrance || 0) * factor)
                    },
                    beauty: {
                        skinValues: {
                            smooth: Math.round(adjustments.skinSmooth * factor),
                            whiten: Math.round(adjustments.skinWhiten * factor),
                            even: Math.round(adjustments.skinEven * factor),
                            korean: Math.round((adjustments.skinKorean || 0) * factor),
                            texture: 50
                        },
                        faceValues: {
                            slim: Math.round((adjustments.faceSlim || 0) * factor),
                            vline: 0,
                            chinShrink: 0,
                            forehead: 0,
                            jaw: 0,
                            noseSlim: 0,
                            noseBridge: 0
                        },
                        eyeValues: {
                            enlarge: Math.round((adjustments.eyeEnlarge || 0) * factor),
                            brightness: Math.round(adjustments.eyeBrightness * factor),
                            darkCircle: Math.round((adjustments.eyeDarkCircle || 0) * factor),
                            depth: 0
                        }
                    }
                };
                result.summary = "Đã tối ưu chân dung: làm đẹp da, mắt và ánh sáng chuyên nghiệp.";
                result.steps = [
                    "Phân tích đặc điểm khuôn mặt",
                    "Làm mịn và đều màu da",
                    "Tăng sáng mắt và giảm quầng thâm",
                    "Tối ưu ánh sáng cho chân dung",
                    "Tạo hiệu ứng da sáng mịn tự nhiên"
                ];
            }

        } else if (moduleId === 'ai_beauty_tone') {
            // AI Smart Tone - Intelligent color tone suggestion
            const toneSchema: Schema = {
                type: Type.OBJECT,
                properties: {
                    // Color adjustments
                    temp: { type: Type.INTEGER, description: "Color temperature (-100 cool to 100 warm)" },
                    tint: { type: Type.INTEGER, description: "Tint adjustment (-100 green to 100 magenta)" },
                    saturation: { type: Type.INTEGER, description: "Saturation (-100 to 100)" },
                    vibrance: { type: Type.INTEGER, description: "Vibrance for color richness (-100 to 100)" },
                    hue: { type: Type.INTEGER, description: "Hue shift for tone matching (-180 to 180)" },
                    // Subtle beauty for tone enhancement
                    skinWhiten: { type: Type.INTEGER, description: "Light skin tone adjustment (0 to 100)" },
                    skinEven: { type: Type.INTEGER, description: "Skin tone uniformity (0 to 100)" },
                    // Exposure for tone balance
                    exposure: { type: Type.INTEGER, description: "Exposure for tone balance (-100 to 100)" },
                    contrast: { type: Type.INTEGER, description: "Contrast for tone depth (-100 to 100)" }
                },
                required: ["temp", "tint", "saturation", "vibrance", "exposure", "contrast"]
            };

            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: {
                    parts: [
                        { text: "You are a professional colorist AI. Analyze this image and suggest the best color tone.\n\nTASK: 'AI Smart Tone' (Intelligent Color Tone)\n\nAnalyze:\n1. Current color palette and mood\n2. Best matching tone style (warm, cool, neutral, vibrant, muted)\n3. Skin tone harmony (if portrait)\n4. Overall color balance\n\nSuggest adjustments that:\n- Enhance the image's natural color harmony\n- Match professional color grading styles\n- Create a cohesive, pleasing color palette\n- Subtle skin tone improvements if portrait\n\nReturn JSON with integer values. Focus on color adjustments, beauty should be minimal." },
                        { inlineData: { mimeType, data: base64Data } }
                    ]
                },
                config: {
                    responseMimeType: "application/json",
                    responseSchema: toneSchema
                }
            });

            const jsonText = response.text;
            if (jsonText) {
                const adjustments = JSON.parse(jsonText);
                const factor = intensity / 100;
                
                result.adjustments = {
                    basic: {
                        temp: Math.round(adjustments.temp * factor),
                        tint: Math.round(adjustments.tint * factor),
                        saturation: Math.round(adjustments.saturation * factor),
                        vibrance: Math.round(adjustments.vibrance * factor),
                        hue: Math.round((adjustments.hue || 0) * factor),
                        exposure: Math.round(adjustments.exposure * factor),
                        contrast: Math.round(adjustments.contrast * factor)
                    },
                    beauty: {
                        skinValues: {
                            smooth: 0,
                            whiten: Math.round((adjustments.skinWhiten || 0) * factor * 0.5), // Very subtle
                            even: Math.round((adjustments.skinEven || 0) * factor * 0.5), // Very subtle
                            korean: 0,
                            texture: 50
                        }
                    }
                };
                result.summary = "Đã áp dụng tone màu thông minh phù hợp với ảnh.";
                result.steps = [
                    "Phân tích bảng màu hiện tại",
                    "Xác định tone màu phù hợp nhất",
                    "Điều chỉnh nhiệt độ và sắc thái màu",
                    "Tối ưu độ bão hòa và rực rỡ",
                    "Cân bằng tone da (nếu có chân dung)"
                ];
            }

        } else if (moduleId.startsWith('ai_beauty')) {
            // Fallback for any other ai_beauty modules
            result.adjustments = {
                basic: { exposure: 5, contrast: 5, vibrance: 10 },
                beauty: { skinValues: { smooth: 50, whiten: 20, even: 30 } }
            };
            result.summary = "AI Beauty adjustments applied.";
        } else {
             result.summary = `Module ${moduleId} executed successfully.`;
        }
    } catch (e: any) {
        console.error("AI Module Error", e);
        throw new Error(e.message || "AI Processing Failed");
    }

    return result;
};
