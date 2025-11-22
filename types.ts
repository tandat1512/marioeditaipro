import React from 'react';

export enum TabType {
  BASIC = 'basic',
  CROP = 'crop',
  TEXT = 'text',
  BEAUTY = 'beauty',
  BODY = 'body',
  FILTER = 'filter',
  EFFECT = 'effect',
  AI_PRO = 'ai_pro'
}

export interface Point {
    x: number; // Percentage 0-100
    y: number; // Percentage 0-100
}

export interface SliderProps {
  label: string;
  value: number;
  min?: number;
  max?: number;
  onChange: (val: number) => void;
}

export interface ToolIconProps {
  icon: React.ComponentType<any>;
  label: string;
  isActive?: boolean;
  onClick: () => void;
}

export interface HistogramData {
  r: number[];
  g: number[];
  b: number[];
  l: number[];
}

export interface TransformValues {
    rotate: number; // 0, 90, 180, 270... (Step rotation)
    rotateFree: number; // Fine rotation slider 1
    flipHorizontal: boolean;
    flipVertical: boolean;
    straighten: number; // Fine rotation slider 2 (-45 to 45 degrees)
    aspectRatio: string; // 'original', 'free', '1:1', '4:5', '16:9', '9:16', '3:4'
    crop: CropData | null;
}

export interface CropData {
    x: number; // Percentage
    y: number; // Percentage
    width: number; // Percentage
    height: number; // Percentage
}

export interface BasicValues {
    exposure: number;
    brightness: number;
    contrast: number;
    highlights: number;
    shadows: number;
    whites: number;
    blacks: number;
    temp: number;
    tint: number;
    vibrance: number;
    saturation: number;
    hue: number;
    grayscale: number;
    sharpen: number;
    blur: number;
    clarity: number;
    texture: number;
    dehaze: number;
    denoise: number;
}

export interface BeautyValues {
    skinMode: 'natural' | 'strong';
    faceMode: 'natural';
    
    // New Skin Architecture
    skinValues: { 
        smooth: number; 
        whiten: number; 
        even: number; 
        korean: number; 
        texture: number; 
    };
    
    // Acne Handling
    acneMode: {
        auto: boolean; // Auto AI Acne Removal
        manualPoints: Point[]; // List of clicks for manual removal
    };

    faceValues: { 
        slim: number; 
        vline: number; 
        chinShrink: number; 
        forehead: number; 
        jaw: number; 
        noseSlim: number; 
        noseBridge: number; 
    };
    eyeValues: { 
        enlarge: number; 
        darkCircle: number; 
        depth: number; 
    };
    eyeMakeup: { 
        lens: string; 
    };
    mouthValues: { 
        smile: number; 
    };
    lipstick: string;
    hairValues: { 
        smooth: number; 
        volume: number; 
        shine: number; 
    };
    hairColor: string;
}

export interface EffectsValues {
    bokeh: { preset: string; intensity: number };
    lightLeak: { preset: string; intensity: number };
    filmGrain: number;
    vignette: number;
    portraitBlur: number;
}

export type FilterCategoryType = 'trending' | 'korean' | 'japanese' | 'pastel' | 'film' | 'bw';

export interface FilterPreset {
  id: string;
  label: string;
  category: FilterCategoryType;
  preview: string;
}

export interface FilterValues {
  selectedCategory: FilterCategoryType;
  selectedPreset: string | null;
  intensity: number;
}

export interface TextLayer {
    id: string;
    text: string;
    x: number; // Percentage 0-100 relative to image width
    y: number; // Percentage 0-100 relative to image height
    fontSize: number;
    fontFamily: string;
    color: string;
    align: 'left' | 'center' | 'right';
    isBold: boolean;
    isItalic: boolean;
    isUnderline: boolean;
    lineHeight: number;
    letterSpacing: number;
    opacity: number;
}

export interface AIPreviewState {
    canvasImage: string | null;
    baseImage: string | null;
    sourceModule: string | null;
}

export interface HistoryState {
    basicValues: BasicValues;
    transformValues: TransformValues;
    effectsValues: EffectsValues;
    textLayers: TextLayer[];
    beautyValues: BeautyValues;
    filterValues: FilterValues;
    aiPreviewState?: AIPreviewState;
}