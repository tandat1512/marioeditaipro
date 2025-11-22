
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Header } from './components/Header';
import { LeftSidebar } from './components/LeftSidebar';
import { Sidebar } from './components/Sidebar';
import { Canvas } from './components/Canvas';
import { TabType, HistogramData, TransformValues, BasicValues, EffectsValues, HistoryState, TextLayer, BeautyValues, FilterValues, CropData } from './types';
import { processImageBasic, calculateAutoSettings, processImageTransform, processImageEffects, processImageFilters, calculateHistogram } from './utils/imageProcessor';
import { runAiProModule, AIProResult, AIProAdjustments, AIPreviewMeta } from './utils/aiProClient';
import { applyBeauty, checkBackendHealth } from './utils/beautyClient';

type AIStatus = 'idle' | 'running' | 'done' | 'error';
type AIPreviewState = { preview: string; mask?: string | null; meta?: AIPreviewMeta | null };
const QUALITY_MODULE_IDS = new Set(['ai_quality_superres', 'ai_quality_restore']);

const serializeHistorySnapshot = (snapshot: HistoryState) => JSON.stringify(snapshot);

function App() {
  const [activeTab, setActiveTab] = useState<TabType>(TabType.BASIC);
  
  // 1. Original uploaded file (Source)
  const [originalImage, setOriginalImage] = useState<string | null>(null);

  // 1.5. Intermediate Image (Rotate/Flip result, before crop) - for displaying full image during crop editing
  const [intermediateImage, setIntermediateImage] = useState<string | null>(null);

  // 2. Transformed Image (Crop/Rotate/Flip result)
  const [transformedImage, setTransformedImage] = useState<string | null>(null);

  // 3. Basic Processed Image (Color/Light result)
  const [basicProcessedImage, setBasicProcessedImage] = useState<string | null>(null);

  // 4. Beauty Processed Image
  const [beautyProcessedImage, setBeautyProcessedImage] = useState<string | null>(null);
  
  // 5. Filter Processed Image
  const [filterProcessedImage, setFilterProcessedImage] = useState<string | null>(null);
  
  // Final Display Image
  const [displayImage, setDisplayImage] = useState<string | null>(null);
  
  // Trigger to reset canvas view (zoom/pan) only on new upload
  const [resetViewTrigger, setResetViewTrigger] = useState<number>(0);
  
  const [filename, setFilename] = useState<string>("Untitled.jpg");
  const [histogramData, setHistogramData] = useState<HistogramData | null>(null);

  // --- STATE ---
  const [transformValues, setTransformValues] = useState<TransformValues>({
      rotate: 0,
      rotateFree: 0,
      flipHorizontal: false,
      flipVertical: false,
      straighten: 0,
      aspectRatio: 'original',
      crop: null
  });

  const [basicValues, setBasicValues] = useState<BasicValues>({
    exposure: 0, brightness: 0, contrast: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0,
    temp: 0, tint: 0, vibrance: 0, saturation: 0, hue: 0, grayscale: 0,
    sharpen: 0, blur: 0, clarity: 0, texture: 0, dehaze: 0, denoise: 0
  });

  const [beautyValues, setBeautyValues] = useState<BeautyValues>({
    skinMode: 'natural', 
    faceMode: 'natural',
    skinValues: { smooth: 0, whiten: 0, even: 0, korean: 0, texture: 50 },
    acneMode: { auto: false, manualPoints: [] },
    faceValues: { slim: 0, vline: 0, chinShrink: 0, forehead: 0, jaw: 0, noseSlim: 0, noseBridge: 0 },
    eyeValues: { enlarge: 0, darkCircle: 0, depth: 0, eyelid: 0 },
    eyeMakeup: { eyeliner: false, lens: 'none' },
    mouthValues: { smile: 0, volume: 0, heart: 0, teethWhiten: 0 },
    lipstick: 'none',
    hairValues: { smooth: 0, volume: 0, shine: 0 },
    hairColor: 'original'
  });
  
  const [isManualAcneMode, setIsManualAcneMode] = useState(false);

  const [filterValues, setFilterValues] = useState<FilterValues>({
    selectedCategory: 'trending',
    selectedPreset: null,
    intensity: 70
  });

  const [effectsValues, setEffectsValues] = useState<EffectsValues>({
    bokeh: { preset: 'soft_pink', intensity: 0 },
    lightLeak: { preset: 'warm_sun', intensity: 0 },
    filmGrain: 0,
    vignette: 0,
    portraitBlur: 0
  });

  const [aiProStatus, setAiProStatus] = useState<Record<string, AIStatus>>({});
  const [aiProErrors, setAiProErrors] = useState<Record<string, string>>({});
  const [aiProInsights, setAiProInsights] = useState<Record<string, AIProResult>>({});
  const [aiProPreviews, setAiProPreviews] = useState<Record<string, AIPreviewState>>({});
  const [aiPreviewCanvasImage, setAiPreviewCanvasImage] = useState<string | null>(null);
  const [aiPreviewBaseImage, setAiPreviewBaseImage] = useState<string | null>(null);
  const [aiPreviewSourceModule, setAiPreviewSourceModule] = useState<string | null>(null);

  const clearAiPreviewCanvas = useCallback(() => {
      setAiPreviewCanvasImage(null);
      setAiPreviewBaseImage(null);
      setAiPreviewSourceModule(null);
  }, []);

  // Text State
  const [textLayers, setTextLayers] = useState<TextLayer[]>([]);
  const [activeTextId, setActiveTextId] = useState<string | null>(null);

  // --- HISTORY (UNDO/REDO) ---
  const [history, setHistory] = useState<HistoryState[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const isUndoRedoAction = useRef(false);
  const historyIndexRef = useRef(historyIndex);
  const skipNextAutoHistoryRef = useRef(false);

  useEffect(() => {
      historyIndexRef.current = historyIndex;
  }, [historyIndex]);

  const buildHistorySnapshot = useCallback((overrides?: Partial<HistoryState>): HistoryState => ({
      basicValues: JSON.parse(JSON.stringify(overrides?.basicValues ?? basicValues)),
      transformValues: JSON.parse(JSON.stringify(overrides?.transformValues ?? transformValues)),
      effectsValues: JSON.parse(JSON.stringify(overrides?.effectsValues ?? effectsValues)),
      textLayers: JSON.parse(JSON.stringify(overrides?.textLayers ?? textLayers)),
      beautyValues: JSON.parse(JSON.stringify(overrides?.beautyValues ?? beautyValues)),
      filterValues: JSON.parse(JSON.stringify(overrides?.filterValues ?? filterValues)),
      aiPreviewState: overrides?.aiPreviewState !== undefined 
          ? JSON.parse(JSON.stringify(overrides.aiPreviewState))
          : {
              canvasImage: aiPreviewCanvasImage,
              baseImage: aiPreviewBaseImage,
              sourceModule: aiPreviewSourceModule
          },
  }), [basicValues, transformValues, effectsValues, textLayers, beautyValues, filterValues, aiPreviewCanvasImage, aiPreviewBaseImage, aiPreviewSourceModule]);

  const pushHistorySnapshot = useCallback((overrides?: Partial<HistoryState>) => {
      if (!originalImage) return;
      const snapshot = buildHistorySnapshot(overrides);
      const snapshotKey = serializeHistorySnapshot(snapshot);

      setHistory(prevHistory => {
          const currentIndex = historyIndexRef.current;
          const truncated = prevHistory.slice(0, currentIndex + 1);
          const last = truncated[truncated.length - 1];
          if (last && serializeHistorySnapshot(last) === snapshotKey) {
              return truncated;
          }
          const nextHistory = [...truncated, snapshot];
          const nextIndex = nextHistory.length - 1;
          historyIndexRef.current = nextIndex;
          setHistoryIndex(nextIndex);
          return nextHistory;
      });
  }, [buildHistorySnapshot, originalImage]);

  const pushHistorySnapshotImmediate = useCallback((overrides?: Partial<HistoryState>) => {
      skipNextAutoHistoryRef.current = true;
      pushHistorySnapshot(overrides);
  }, [pushHistorySnapshot]);

  // Monitor changes and save to history with debounce
  useEffect(() => {
      if (!originalImage || isUndoRedoAction.current) {
          isUndoRedoAction.current = false;
          return;
      }

      if (skipNextAutoHistoryRef.current) {
          skipNextAutoHistoryRef.current = false;
          return;
      }

      const timer = setTimeout(() => {
          pushHistorySnapshot();
      }, 400);

      return () => clearTimeout(timer);
  }, [basicValues, transformValues, effectsValues, textLayers, beautyValues, filterValues, originalImage, pushHistorySnapshot]);

  // Undo Handler
  const handleUndo = useCallback(() => {
      if (historyIndex > 0 && originalImage) {
          isUndoRedoAction.current = true;
          const prevIndex = historyIndex - 1;
          const prevState = history[prevIndex];
          
          if (prevState) {
            setBasicValues(prevState.basicValues);
            setTransformValues(prevState.transformValues);
            setEffectsValues(prevState.effectsValues);
            setTextLayers(prevState.textLayers);
            setBeautyValues(prevState.beautyValues);
            setFilterValues(prevState.filterValues);
            if (prevState.aiPreviewState) {
                setAiPreviewCanvasImage(prevState.aiPreviewState.canvasImage);
                setAiPreviewBaseImage(prevState.aiPreviewState.baseImage);
                setAiPreviewSourceModule(prevState.aiPreviewState.sourceModule);
            } else {
                clearAiPreviewCanvas();
            }
            setHistoryIndex(prevIndex);
            historyIndexRef.current = prevIndex;
          }
      }
  }, [historyIndex, originalImage, history, clearAiPreviewCanvas]);

  // Redo Handler
  const handleRedo = useCallback(() => {
      if (historyIndex < history.length - 1 && originalImage) {
          isUndoRedoAction.current = true;
          const nextIndex = historyIndex + 1;
          const nextState = history[nextIndex];
          
          if (nextState) {
            setBasicValues(nextState.basicValues);
            setTransformValues(nextState.transformValues);
            setEffectsValues(nextState.effectsValues);
            setTextLayers(nextState.textLayers);
            setBeautyValues(nextState.beautyValues);
            setFilterValues(nextState.filterValues);
            if (nextState.aiPreviewState) {
                setAiPreviewCanvasImage(nextState.aiPreviewState.canvasImage);
                setAiPreviewBaseImage(nextState.aiPreviewState.baseImage);
                setAiPreviewSourceModule(nextState.aiPreviewState.sourceModule);
            } else {
                clearAiPreviewCanvas();
            }
            setHistoryIndex(nextIndex);
            historyIndexRef.current = nextIndex;
          }
      }
  }, [historyIndex, history, originalImage, clearAiPreviewCanvas]);

  useEffect(() => {
      const shouldSkipGlobalUndo = (target: EventTarget | null) => {
          if (!target) return false;
          if (!(target instanceof HTMLElement)) return false;
          if (target.isContentEditable) return true;
          const tag = target.tagName?.toLowerCase();
          if (tag === 'textarea' || tag === 'select') {
              return true;
          }
          if (tag === 'input') {
              const input = target as HTMLInputElement;
              const type = (input.type || '').toLowerCase();
              const textualTypes = ['text', 'search', 'url', 'tel', 'password', 'email', 'number'];
              if (!type || textualTypes.includes(type)) {
                  return true;
              }
          }
          return false;
      };

      const handleKeyDown = (event: KeyboardEvent) => {
          if (!originalImage) return;
          if (!(event.ctrlKey || event.metaKey)) return;
          if (event.key.toLowerCase() !== 'z') return;
          if (shouldSkipGlobalUndo(event.target)) {
              return;
          }

          event.preventDefault();
          if (event.shiftKey) {
              handleRedo();
          } else {
              handleUndo();
          }
      };

      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleUndo, handleRedo, originalImage]);

  // --- HANDLERS ---

  const handleImageUpload = (file: File) => {
    const url = URL.createObjectURL(file);
    setOriginalImage(url);
    setIntermediateImage(url);
    setTransformedImage(url);
    setBasicProcessedImage(url);
    setBeautyProcessedImage(url);
    setFilterProcessedImage(url);
    setDisplayImage(url);
    setFilename(file.name);
    setResetViewTrigger(Date.now());
    
    // Reset states
    const initialTransform = { rotate: 0, rotateFree: 0, flipHorizontal: false, flipVertical: false, straighten: 0, aspectRatio: 'original', crop: null };
    const initialBasic = {
        exposure: 0, brightness: 0, contrast: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0,
        temp: 0, tint: 0, vibrance: 0, saturation: 0, hue: 0, grayscale: 0,
        sharpen: 0, blur: 0, clarity: 0, texture: 0, dehaze: 0, denoise: 0
    };
    const initialFilter: FilterValues = { selectedCategory: 'trending', selectedPreset: null, intensity: 70 };
    const initialEffects: EffectsValues = { 
        bokeh: { preset: 'soft_pink', intensity: 0 },
        lightLeak: { preset: 'warm_sun', intensity: 0 },
        filmGrain: 0,
        vignette: 0,
        portraitBlur: 0
    };
    const initialTexts: TextLayer[] = [];
    const initialBeauty: BeautyValues = {
        skinMode: 'natural', 
        faceMode: 'natural',
        skinValues: { smooth: 0, whiten: 0, even: 0, korean: 0, texture: 50 },
        acneMode: { auto: false, manualPoints: [] },
        faceValues: { slim: 0, vline: 0, chinShrink: 0, forehead: 0, jaw: 0, noseSlim: 0, noseBridge: 0 },
        eyeValues: { enlarge: 0, darkCircle: 0, depth: 0, eyelid: 0 },
        eyeMakeup: { eyeliner: false, lens: 'none' },
        mouthValues: { smile: 0, volume: 0, heart: 0, teethWhiten: 0 },
        lipstick: 'none',
        hairValues: { smooth: 0, volume: 0, shine: 0 },
        hairColor: 'original'
    };
    
    setTransformValues(initialTransform);
    setBasicValues(initialBasic);
    setFilterValues(initialFilter);
    setEffectsValues(initialEffects);
    setTextLayers(initialTexts);
    setBeautyValues(initialBeauty);
    setActiveTextId(null);
    
    // Init History
    const initialState: HistoryState = { 
        basicValues: initialBasic, 
        transformValues: initialTransform, 
        effectsValues: initialEffects, 
        textLayers: initialTexts, 
        beautyValues: initialBeauty,
        filterValues: initialFilter,
        aiPreviewState: {
            canvasImage: null,
            baseImage: null,
            sourceModule: null
        }
    };
    setHistory([initialState]);
    setHistoryIndex(0);
    historyIndexRef.current = 0;
    skipNextAutoHistoryRef.current = false;
    setAiProStatus({});
    setAiProInsights({});
    setAiProPreviews({});
    clearAiPreviewCanvas();
  };

  const handleClearImage = () => {
      setOriginalImage(null);
      setIntermediateImage(null);
      setTransformedImage(null);
      setBasicProcessedImage(null);
      setBeautyProcessedImage(null);
      setFilterProcessedImage(null);
      setDisplayImage(null);
      setFilename("Untitled.jpg");
      setHistogramData(null);
      setHistory([]);
      setHistoryIndex(-1);
      historyIndexRef.current = -1;
      skipNextAutoHistoryRef.current = false;
      setResetViewTrigger(0);
      setTextLayers([]);
      setActiveTextId(null);
      setFilterValues({ selectedCategory: 'trending', selectedPreset: null, intensity: 70 });
      setEffectsValues({
        bokeh: { preset: 'soft_pink', intensity: 0 },
        lightLeak: { preset: 'warm_sun', intensity: 0 },
        filmGrain: 0,
        vignette: 0,
        portraitBlur: 0
      });
      setAiProStatus({});
      setAiProInsights({});
      setAiProPreviews({});
      clearAiPreviewCanvas();
  };

  // Text Handlers
  const handleAddText = () => {
      const newText: TextLayer = {
          id: Date.now().toString(),
          text: 'Nhập văn bản',
          x: 50,
          y: 50,
          fontSize: 40,
          fontFamily: 'Inter',
          color: '#ffffff',
          align: 'center',
          isBold: false,
          isItalic: false,
          isUnderline: false,
          lineHeight: 1.2,
          letterSpacing: 0,
          opacity: 100
      };
      const nextLayers = [...textLayers, newText];
      setTextLayers(nextLayers);
      setActiveTextId(newText.id);
      pushHistorySnapshotImmediate({ textLayers: nextLayers });
  };

  const handleUpdateText = (id: string, updates: Partial<TextLayer>) => {
      setTextLayers(prev => prev.map(layer => layer.id === id ? { ...layer, ...updates } : layer));
  };
  
  const handleRemoveText = (id: string) => {
      const nextLayers = textLayers.filter(l => l.id !== id);
      setTextLayers(nextLayers);
      if (activeTextId === id) setActiveTextId(null);
      pushHistorySnapshotImmediate({ textLayers: nextLayers });
  };

  const handleAutoAdjust = async () => {
      if (!transformedImage) return;
      try {
          const recommendedSettings = await calculateAutoSettings(transformedImage);
          setBasicValues(recommendedSettings);
      } catch (error) {
          console.error("Auto adjust failed:", error);
      }
  };

  const applyAiAdjustments = useCallback((adjustments?: AIProAdjustments | null) => {
      if (!adjustments) return;

      if (adjustments.basic) {
          setBasicValues(prev => ({ ...prev, ...adjustments.basic }));
      }

      if (adjustments.beauty) {
          setBeautyValues(prev => {
              const next: BeautyValues = { ...prev };
              if (adjustments.beauty?.skinValues) {
                  next.skinValues = { ...prev.skinValues, ...adjustments.beauty.skinValues };
              }
              if (adjustments.beauty?.faceValues) {
                  next.faceValues = { ...prev.faceValues, ...adjustments.beauty.faceValues };
              }
              if (adjustments.beauty?.eyeValues) {
                  next.eyeValues = { ...prev.eyeValues, ...adjustments.beauty.eyeValues };
              }
              if (adjustments.beauty?.mouthValues) {
                  next.mouthValues = { ...prev.mouthValues, ...adjustments.beauty.mouthValues };
              }
              if (adjustments.beauty?.hairValues) {
                  next.hairValues = { ...prev.hairValues, ...adjustments.beauty.hairValues };
              }
              if (adjustments.beauty?.lipstick) {
                  next.lipstick = adjustments.beauty.lipstick;
              }
              return next;
          });
      }

      if (adjustments.filters) {
          setFilterValues(prev => ({ ...prev, ...adjustments.filters }));
      }

      if (adjustments.effects) {
          setEffectsValues(prev => ({ ...prev, ...adjustments.effects }));
      }
  }, []);

  const handleAIProAction = async (featureId: string, payload: { intensity: number; options?: any; referenceImageFile?: File | null }) => {
    // We use beautyProcessedImage as source for most enhancements to include previous edits
    const sourceForAI = beautyProcessedImage || transformedImage || originalImage;
    if (!sourceForAI) return;
    
    setAiProStatus(prev => ({ ...prev, [featureId]: 'running' }));
    setAiProErrors(prev => { const next = { ...prev }; delete next[featureId]; return next; });

    try {
        const result = await runAiProModule(sourceForAI, featureId, payload.intensity, payload.options, { referenceImageFile: payload.referenceImageFile || null });
        
        setAiProInsights(prev => ({ ...prev, [featureId]: result }));
        
        if (result.previewImage) {
             setAiProPreviews(prev => ({
                 ...prev,
                 [featureId]: {
                     preview: result.previewImage!,
                     mask: result.maskImage,
                     meta: result.previewMeta
                 }
             }));
        }

        if (result.adjustments) {
            applyAiAdjustments(result.adjustments);
        }

        setAiProStatus(prev => ({ ...prev, [featureId]: 'done' }));

    } catch (error: any) {
        console.error("AI Action Failed", error);
        setAiProErrors(prev => ({ ...prev, [featureId]: error.message || "Processing failed" }));
        setAiProStatus(prev => ({ ...prev, [featureId]: 'error' }));
    }
  };

  const handleApplyAiPreview = async (moduleId: string) => {
      const preview = aiProPreviews[moduleId];
      if (preview?.preview) {
          pushHistorySnapshotImmediate();
          
          // Replacing original image with AI result
          setOriginalImage(preview.preview);
          // Reset transformations because usually AI output is final or a new base
          // However, for style transfer we might want to keep cropping. 
          // For Background removal, we might want to keep cropping.
          // But if we replace 'original', the 'crop' logic will re-apply to the NEW image.
          // If the new image is already cropped?
          // Let's assume standard AI flow here: New Base.
          setTransformValues({ rotate: 0, rotateFree: 0, flipHorizontal: false, flipVertical: false, straighten: 0, aspectRatio: 'original', crop: null });
          setBasicValues({ exposure: 0, brightness: 0, contrast: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0, temp: 0, tint: 0, vibrance: 0, saturation: 0, hue: 0, grayscale: 0, sharpen: 0, blur: 0, clarity: 0, texture: 0, dehaze: 0, denoise: 0 });
          setBeautyValues({ skinMode: 'natural', faceMode: 'natural', skinValues: { smooth: 0, whiten: 0, even: 0, korean: 0, texture: 50 }, acneMode: { auto: false, manualPoints: [] }, faceValues: { slim: 0, vline: 0, chinShrink: 0, forehead: 0, jaw: 0, noseSlim: 0, noseBridge: 0 }, eyeValues: { enlarge: 0, darkCircle: 0, depth: 0, eyelid: 0 }, eyeMakeup: { eyeliner: false, lens: 'none' }, mouthValues: { smile: 0, volume: 0, heart: 0, teethWhiten: 0 }, lipstick: 'none', hairValues: { smooth: 0, volume: 0, shine: 0 }, hairColor: 'original' });
          setFilterValues({ selectedCategory: 'trending', selectedPreset: null, intensity: 70 });
          
          handleDismissAiPreview(moduleId);
      }
  };

  const handleDismissAiPreview = (moduleId: string) => {
      setAiProPreviews(prev => {
          const next = { ...prev };
          delete next[moduleId];
          return next;
      });
      setAiProStatus(prev => ({ ...prev, [moduleId]: 'idle' }));
  };

  const handleAIImageGenerated = (dataUrl: string) => {
      pushHistorySnapshotImmediate();
      setOriginalImage(dataUrl);
      setIntermediateImage(dataUrl);
      setTransformedImage(dataUrl);
      setBasicProcessedImage(dataUrl);
      setBeautyProcessedImage(dataUrl);
      setFilterProcessedImage(dataUrl);
      setDisplayImage(dataUrl);
      // Reset pipeline
      setTransformValues({ rotate: 0, rotateFree: 0, flipHorizontal: false, flipVertical: false, straighten: 0, aspectRatio: 'original', crop: null });
      setBasicValues({ exposure: 0, brightness: 0, contrast: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0, temp: 0, tint: 0, vibrance: 0, saturation: 0, hue: 0, grayscale: 0, sharpen: 0, blur: 0, clarity: 0, texture: 0, dehaze: 0, denoise: 0 });
  };

  // --- IMAGE PROCESSING PIPELINE ---
  
  // 1. Transform
  useEffect(() => {
      if (!originalImage) return;
      const runTransform = async () => {
           const res = await processImageTransform(originalImage, transformValues);
           setTransformedImage(res);
           
           if (activeTab === TabType.CROP) {
               const noCropValues = { ...transformValues, crop: null };
               const resNoCrop = await processImageTransform(originalImage, noCropValues);
               setIntermediateImage(resNoCrop);
           }
      };
      runTransform();
  }, [originalImage, transformValues, activeTab]);

  // 2. Basic
  useEffect(() => {
      if (!transformedImage) return;
      processImageBasic(transformedImage, basicValues).then(setBasicProcessedImage);
  }, [transformedImage, basicValues]);

  // Backend availability state
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  
  // Check backend health on mount
  useEffect(() => {
      checkBackendHealth().then(setBackendAvailable);
  }, []);

  // 3. Beauty - Use backend API if available, fallback to client-side
  useEffect(() => {
      if (!basicProcessedImage) return;
      
      const t = setTimeout(async () => {
          try {
              // Check if we should use backend (only if backend is confirmed available)
              if (backendAvailable === true) {
                  try {
                      const result = await applyBeauty(basicProcessedImage, beautyValues);
                      setBeautyProcessedImage(result.image);
                      return;
                  } catch (error) {
                      console.warn('Backend API failed, falling back to client-side:', error);
                      // Fallback to client-side if backend fails
                      setBackendAvailable(false);
                  }
              }
              
              // Fallback to client-side processing
              const { processImageBeauty } = await import('./utils/imageProcessor');
              const result = await processImageBeauty(basicProcessedImage, beautyValues);
              setBeautyProcessedImage(result);
          } catch (error) {
              console.error('Beauty processing failed:', error);
              // On error, just use the basic processed image
              setBeautyProcessedImage(basicProcessedImage);
          }
      }, 100);
      
      return () => clearTimeout(t);
  }, [basicProcessedImage, beautyValues, backendAvailable]);
  
  // 4. Filters
  useEffect(() => {
      if (!beautyProcessedImage) return;
      processImageFilters(beautyProcessedImage, filterValues).then(setFilterProcessedImage);
  }, [beautyProcessedImage, filterValues]);
  
  // 5. Effects
  useEffect(() => {
      if (!filterProcessedImage) return;
      processImageEffects(filterProcessedImage, effectsValues).then(setDisplayImage);
  }, [filterProcessedImage, effectsValues]);
  
  // 6. Histogram
  useEffect(() => {
      if (!displayImage) {
          setHistogramData(null);
          return;
      }
      const t = setTimeout(() => {
          calculateHistogram(displayImage).then(setHistogramData).catch(console.error);
      }, 200);
      return () => clearTimeout(t);
  }, [displayImage]);


  // --- RENDER ---

  return (
    <div className="flex h-screen flex-col bg-[#09090b] text-white overflow-hidden">
      <Header filename={filename} hasImage={!!originalImage} displayImage={displayImage} />
      
      <div className="flex flex-1 overflow-hidden">
        <LeftSidebar 
            activeTab={activeTab} 
            onSelect={setActiveTab} 
            hasImage={!!originalImage} 
            onRemove={handleClearImage}
        />
        
        <Canvas 
            image={displayImage}
            compareImage={originalImage}
            intermediateImage={intermediateImage}
            onUpload={handleImageUpload}
            onClear={handleClearImage}
            onUndo={handleUndo}
            onRedo={handleRedo}
            canUndo={historyIndex > 0}
            canRedo={historyIndex < history.length - 1}
            resetViewTrigger={resetViewTrigger}
            textLayers={textLayers}
            activeTextId={activeTextId}
            onTextClick={setActiveTextId}
            onTextUpdate={handleUpdateText}
            onTextRemove={handleRemoveText}
            isManualAcneMode={isManualAcneMode}
            onCanvasClick={(x, y) => {
                 // Add manual acne point
                 setBeautyValues(prev => ({
                     ...prev,
                     acneMode: {
                         ...prev.acneMode,
                         manualPoints: [...prev.acneMode.manualPoints, { x, y }]
                     }
                 }));
            }}
            transformValues={transformValues}
            onTransformChange={setTransformValues}
            activeTab={activeTab}
        />

        <Sidebar 
            activeTab={activeTab}
            histogramData={histogramData}
            basicValues={basicValues}
            onBasicChange={setBasicValues}
            onAutoAdjust={handleAutoAdjust}
            transformValues={transformValues}
            onTransformChange={setTransformValues}
            effectsValues={effectsValues}
            onEffectsChange={setEffectsValues}
            textLayers={textLayers}
            activeTextId={activeTextId}
            onAddText={handleAddText}
            onUpdateText={handleUpdateText}
            beautyValues={beautyValues}
            onBeautyChange={setBeautyValues}
            filterValues={filterValues}
            onFilterChange={setFilterValues}
            isManualAcneMode={isManualAcneMode}
            toggleManualAcneMode={() => setIsManualAcneMode(!isManualAcneMode)}
            // AI Props
            onAIProAction={handleAIProAction}
            aiProStatus={aiProStatus}
            aiProErrors={aiProErrors}
            aiProInsights={aiProInsights}
            aiProPreviews={aiProPreviews}
            onApplyAiPreview={handleApplyAiPreview}
            onDismissAiPreview={handleDismissAiPreview}
            currentImage={displayImage}
            onAIImageGenerated={handleAIImageGenerated}
        />
      </div>
    </div>
  );
}

export default App;
