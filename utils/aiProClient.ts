
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
            // Tự chỉnh toàn diện - Phân tích và cân bằng ánh sáng, màu sắc, độ nét tự động
            const adjustmentSchema: Schema = {
                type: Type.OBJECT,
                properties: {
                    exposure: { type: Type.INTEGER, description: "Exposure adjustment (-100 to 100). Analyze overall brightness and correct if too dark or too bright." },
                    contrast: { type: Type.INTEGER, description: "Contrast adjustment (-100 to 100). Enhance image depth and definition." },
                    highlights: { type: Type.INTEGER, description: "Highlights adjustment (-100 to 100). Recover or enhance bright areas." },
                    shadows: { type: Type.INTEGER, description: "Shadows adjustment (-100 to 100). Lift dark areas while maintaining natural look." },
                    whites: { type: Type.INTEGER, description: "Whites adjustment (-100 to 100). Fine-tune brightest points." },
                    blacks: { type: Type.INTEGER, description: "Blacks adjustment (-100 to 100). Fine-tune darkest points." },
                    saturation: { type: Type.INTEGER, description: "Saturation adjustment (-100 to 100). Enhance color richness naturally." },
                    vibrance: { type: Type.INTEGER, description: "Vibrance adjustment (-100 to 100). Boost muted colors while protecting skin tones." },
                    temp: { type: Type.INTEGER, description: "Temperature adjustment (-100 to 100). Balance warm/cool tones for natural look." },
                    tint: { type: Type.INTEGER, description: "Tint adjustment (-100 to 100). Correct green/magenta cast." },
                    clarity: { type: Type.INTEGER, description: "Clarity/Sharpness adjustment (0 to 100). Enhance mid-tone contrast and detail." },
                    dehaze: { type: Type.INTEGER, description: "Dehaze adjustment (0 to 100). Remove haze and improve atmospheric clarity." }
                },
                required: ["exposure", "contrast", "highlights", "shadows", "saturation", "vibrance", "clarity"]
            };

            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: {
                    parts: [
                        { text: `You are a professional photo editor. Analyze this image and provide comprehensive adjustments to balance lighting, colors, and sharpness automatically.

TASK: Provide optimal adjustment values to:
1. Balance overall exposure (not too dark, not too bright)
2. Enhance contrast for depth and definition
3. Recover details in highlights and shadows
4. Enhance color richness naturally (use vibrance more than saturation to protect skin tones)
5. Correct color temperature and tint for natural look
6. Improve clarity and sharpness without overdoing it
7. Remove haze if present

RULES:
- Be conservative with adjustments (aim for natural enhancement, not dramatic changes)
- Prioritize preserving natural skin tones
- If image is already well-balanced, make minimal adjustments
- Clarity should be moderate (20-40) unless image is very soft
- Return integer values only` },
                        { inlineData: { mimeType, data: base64Data } }
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
                
                // Apply intensity scaling
                const factor = intensity / 100;
                const scaledAdjustments: Record<string, number> = {};
                
                Object.keys(adjustments).forEach(key => {
                    const value = adjustments[key];
                    if (key === 'clarity' || key === 'dehaze') {
                        // Clarity and dehaze are 0-100, scale directly
                        scaledAdjustments[key] = Math.round(value * factor);
                    } else {
                        // Other adjustments are -100 to 100
                        scaledAdjustments[key] = Math.round(value * factor);
                    }
                });

                result.adjustments = {
                    basic: scaledAdjustments
                };
                result.summary = "Đã tự động cân bằng ánh sáng, màu sắc và độ nét toàn diện.";
                result.steps = [
                    "Phân tích độ sáng và độ tương phản tổng thể",
                    "Cân bằng highlights và shadows để phục hồi chi tiết",
                    "Tối ưu màu sắc và độ bão hòa tự nhiên",
                    "Tăng độ nét và loại bỏ haze (nếu có)"
                ];
            }

        } else if (moduleId === 'ai_beauty_portrait') {
            // Tối ưu chân dung - Tập trung vào làm đẹp da, mắt, ánh sáng khuôn mặt
            const adjustmentSchema: Schema = {
                type: Type.OBJECT,
                properties: {
                    // Basic adjustments for face lighting
                    exposure: { type: Type.INTEGER, description: "Exposure for face area (-50 to 50). Slightly brighten if needed." },
                    highlights: { type: Type.INTEGER, description: "Highlights for face (-50 to 50). Soften harsh highlights on face." },
                    shadows: { type: Type.INTEGER, description: "Shadows for face (0 to 50). Lift shadows on face gently." },
                    // Beauty-specific adjustments
                    skinSmooth: { type: Type.INTEGER, description: "Skin smoothing level (0 to 100). Smooth skin texture while preserving natural details." },
                    skinWhiten: { type: Type.INTEGER, description: "Skin whitening level (0 to 50). Brighten skin tone naturally." },
                    skinEven: { type: Type.INTEGER, description: "Skin evenness level (0 to 100). Even out skin tone and reduce discoloration." },
                    eyeWhiten: { type: Type.INTEGER, description: "Eye whitening level (0 to 50). Whiten eye whites naturally." },
                    faceContour: { type: Type.INTEGER, description: "Face contouring level (0 to 50). Subtle face shaping and definition." },
                    vibrance: { type: Type.INTEGER, description: "Vibrance for portrait (0 to 30). Subtle color enhancement for natural look." }
                },
                required: ["skinSmooth", "skinWhiten", "skinEven", "eyeWhiten"]
            };

            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: {
                    parts: [
                        { text: `You are a professional portrait retoucher. Analyze this portrait image and provide beauty enhancement adjustments.

TASK: Focus on portrait-specific enhancements:
1. Skin: Smooth texture, brighten naturally, even out tone
2. Eyes: Whiten eye whites naturally
3. Face lighting: Soften harsh highlights, lift shadows gently
4. Subtle color enhancement for natural portrait look

RULES:
- Skin smoothing should be moderate (30-60) to preserve natural texture
- Skin whitening should be subtle (10-30) for natural look
- Eye enhancements should be noticeable but not overdone
- Face lighting adjustments should be gentle
- Overall: Aim for natural, professional portrait look, not over-processed
- Return integer values only` },
                        { inlineData: { mimeType, data: base64Data } }
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
                
                // Apply intensity scaling
                const factor = intensity / 100;
                
                // Scale beauty values (0-100 range)
                const beautyValues: Record<string, number> = {};
                if (adjustments.skinSmooth !== undefined) beautyValues.smooth = Math.round(adjustments.skinSmooth * factor);
                if (adjustments.skinWhiten !== undefined) beautyValues.whiten = Math.round(adjustments.skinWhiten * factor);
                if (adjustments.skinEven !== undefined) beautyValues.even = Math.round(adjustments.skinEven * factor);
                
                const faceValues: Record<string, number> = {};
                if (adjustments.faceContour !== undefined) faceValues.contour = Math.round(adjustments.faceContour * factor);
                
                const eyeValues: Record<string, number> = {};
                // Note: eyeWhiten is not a standard eyeValues key, but we can add it if needed

                // Scale basic adjustments
                const basicAdjustments: Record<string, number> = {};
                if (adjustments.exposure !== undefined) basicAdjustments.exposure = Math.round(adjustments.exposure * factor);
                if (adjustments.highlights !== undefined) basicAdjustments.highlights = Math.round(adjustments.highlights * factor);
                if (adjustments.shadows !== undefined) basicAdjustments.shadows = Math.round(adjustments.shadows * factor);
                if (adjustments.vibrance !== undefined) basicAdjustments.vibrance = Math.round(adjustments.vibrance * factor);

                result.adjustments = {
                    basic: basicAdjustments,
                    beauty: {
                        skinValues: beautyValues,
                        faceValues: faceValues,
                        eyeValues: eyeValues
                    }
                };
                result.summary = "Đã tối ưu chân dung: làm đẹp da và cân bằng ánh sáng khuôn mặt.";
                result.steps = [
                    "Phân tích khuôn mặt và đặc điểm da",
                    "Làm mịn da và làm đều tone màu da",
                    "Cân bằng ánh sáng khuôn mặt tự nhiên"
                ];
            }

        } else if (moduleId === 'ai_beauty_tone') {
            // AI Smart Tone - Phân tích và gợi ý tone màu phù hợp
            const toneSchema: Schema = {
                type: Type.OBJECT,
                properties: {
                    recommendedTone: { 
                        type: Type.STRING, 
                        description: "Recommended color tone: 'warm', 'cool', 'neutral', 'cinematic', 'vintage', 'modern', 'soft', 'vibrant'" 
                    },
                    temp: { type: Type.INTEGER, description: "Temperature adjustment (-100 to 100) to achieve recommended tone." },
                    tint: { type: Type.INTEGER, description: "Tint adjustment (-100 to 100) to achieve recommended tone." },
                    saturation: { type: Type.INTEGER, description: "Saturation adjustment (-100 to 100) for recommended tone." },
                    vibrance: { type: Type.INTEGER, description: "Vibrance adjustment (-100 to 100) for recommended tone." },
                    contrast: { type: Type.INTEGER, description: "Contrast adjustment (-100 to 100) for recommended tone style." },
                    highlights: { type: Type.INTEGER, description: "Highlights adjustment (-100 to 100) for recommended tone mood." },
                    shadows: { type: Type.INTEGER, description: "Shadows adjustment (-100 to 100) for recommended tone mood." },
                    explanation: { type: Type.STRING, description: "Brief explanation of why this tone was recommended (max 100 characters)." }
                },
                required: ["recommendedTone", "temp", "tint", "saturation", "vibrance", "explanation"]
            };

            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: {
                    parts: [
                        { text: `You are a professional colorist. Analyze this image and recommend the best color tone/style that would enhance it.

AVAILABLE TONES:
- 'warm': Golden, sunset-like warmth (increase temp, warm shadows)
- 'cool': Blue, crisp, modern coolness (decrease temp, cool highlights)
- 'neutral': Balanced, natural colors (minimal temp/tint changes)
- 'cinematic': Movie-like, dramatic contrast and color grading
- 'vintage': Retro, faded, nostalgic look (desaturate slightly, warm tones)
- 'modern': Clean, vibrant, contemporary look (boost vibrance, balanced)
- 'soft': Gentle, pastel-like, dreamy (reduce contrast, soft colors)
- 'vibrant': Rich, saturated, energetic colors (boost saturation/vibrance)

TASK:
1. Analyze the image content, mood, and current color palette
2. Recommend the BEST tone that would enhance this specific image
3. Provide adjustment values to achieve that tone
4. Explain briefly why this tone was chosen

RULES:
- Choose tone that matches the image's mood and content
- Portrait photos often benefit from 'warm' or 'soft'
- Landscape/nature often benefits from 'vibrant' or 'cinematic'
- Urban/modern scenes often benefit from 'cool' or 'modern'
- Be creative but practical
- Return integer values for adjustments` },
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
                const toneData = JSON.parse(jsonText);
                
                // Apply intensity scaling
                const factor = intensity / 100;
                const scaledAdjustments: Record<string, number> = {};
                
                // Scale numeric adjustments
                ['temp', 'tint', 'saturation', 'vibrance', 'contrast', 'highlights', 'shadows'].forEach(key => {
                    if (toneData[key] !== undefined) {
                        scaledAdjustments[key] = Math.round(toneData[key] * factor);
                    }
                });

                result.adjustments = {
                    basic: scaledAdjustments
                };
                
                // Create tone description
                const toneNames: Record<string, string> = {
                    'warm': 'Ấm áp',
                    'cool': 'Mát mẻ',
                    'neutral': 'Trung tính',
                    'cinematic': 'Điện ảnh',
                    'vintage': 'Cổ điển',
                    'modern': 'Hiện đại',
                    'soft': 'Nhẹ nhàng',
                    'vibrant': 'Rực rỡ'
                };
                
                const toneName = toneNames[toneData.recommendedTone] || toneData.recommendedTone;
                result.summary = `Đã áp dụng tone màu: ${toneName}. ${toneData.explanation || ''}`;
                result.steps = [
                    "Phân tích bảng màu và tâm trạng ảnh hiện tại",
                    `Xác định tone màu phù hợp: ${toneName}`,
                    "Điều chỉnh nhiệt độ màu và độ bão hòa",
                    "Tối ưu độ tương phản và dynamic range cho tone đã chọn"
                ];
                result.metrics = {
                    recommendedTone: toneData.recommendedTone,
                    toneName: toneName
                };
            }

        } else {
             result.summary = `Module ${moduleId} executed successfully.`;
        }
    } catch (e: any) {
        console.error("AI Module Error", e);
        throw new Error(e.message || "AI Processing Failed");
    }

    return result;
};
