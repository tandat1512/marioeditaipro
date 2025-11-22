
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { 
  ChevronDown, Play, Check, RotateCcw, RotateCw, FlipHorizontal, FlipVertical, BarChart3, Type, AlignLeft, AlignCenter, AlignRight, Bold, Italic, Underline, Wand2, Eye, Smile, Sun, Sparkles, Film, Waves, Scan, Fingerprint, Loader2
} from 'lucide-react';
import { TabType, HistogramData, TransformValues, BasicValues, EffectsValues, TextLayer, BeautyValues, FilterValues, FilterCategoryType, FilterPreset } from '../types';
import { generateSceneImage, DEFAULT_GEMINI_SCENE_MODEL } from '../utils/geminiImage';
import { SliderControl } from './SliderControl';
import type { AIProResult, AIPreviewMeta } from '../utils/aiProClient';

// Types for props
interface SidebarProps {
    activeTab: TabType;
    histogramData: HistogramData | null;
    // Basic
    basicValues: BasicValues; 
    onBasicChange: (val: any) => void;
    onAutoAdjust?: () => void;
    // Transform
    transformValues: TransformValues;
    onTransformChange: (val: TransformValues) => void;
    // Effects
    effectsValues: EffectsValues;
    onEffectsChange: (val: React.SetStateAction<EffectsValues>) => void;
    // Text
    textLayers?: TextLayer[];
    activeTextId?: string | null;
    onAddText?: () => void;
    onUpdateText?: (id: string, updates: Partial<TextLayer>) => void;
    // Beauty
    beautyValues?: BeautyValues;
    onBeautyChange?: (val: React.SetStateAction<BeautyValues>) => void;
    onApplyBeautyPreset?: (mode: 'natural' | 'strong') => void;
    // Manual Acne State
    isManualAcneMode?: boolean;
    toggleManualAcneMode?: () => void;
    // Filters
    filterValues: FilterValues;
    onFilterChange: (val: React.SetStateAction<FilterValues>) => void;
    onAIProAction?: (featureId: string, payload: { intensity: number; options?: Record<string, unknown>; referenceImageFile?: File | null }) => void | Promise<void>;
    aiProStatus?: Record<string, AIStatus>;
    aiProErrors?: Record<string, string>;
    aiProInsights?: Record<string, AIProResult>;
    aiProPreviews?: AIPreviewMap;
    onApplyAiPreview?: (moduleId: string) => void | Promise<void>;
    onDismissAiPreview?: (moduleId: string) => void;
    currentImage?: string | null;
    onAIImageGenerated?: (imageDataUrl: string) => void;
}

// --- FILTER DATA DEFINITIONS ---
const FILTER_CATEGORIES: { id: FilterCategoryType; label: string }[] = [
    { id: 'trending', label: 'Trending' },
    { id: 'korean', label: 'Hàn Quốc' },
    { id: 'japanese', label: 'Nhật Bản' },
    { id: 'pastel', label: 'Pastel' },
    { id: 'film', label: 'Film' },
    { id: 'bw', label: 'B&W' },
];

const FILTERS_DATA: FilterPreset[] = [
    { id: 'trend_glow', label: 'Glow Muse', category: 'trending', preview: 'linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)' },
    { id: 'trend_city', label: 'City Pop', category: 'trending', preview: 'linear-gradient(135deg, #f6d365 0%, #fda085 100%)' },
    { id: 'kor_peach', label: 'Peach Skin', category: 'korean', preview: 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)' },
    { id: 'kor_beige', label: 'Beige Glow', category: 'korean', preview: 'linear-gradient(135deg, #eacda3 0%, #d6ae7b 100%)' },
    { id: 'kor_snow', label: 'Snow White', category: 'korean', preview: 'linear-gradient(135deg, #dfe9f3 0%, #ffffff 100%)' },
    { id: 'jp_fuji', label: 'Fuji Film', category: 'japanese', preview: 'linear-gradient(135deg, #96fbc4 0%, #f9f586 100%)' },
    { id: 'jp_street', label: 'Tokyo Night', category: 'japanese', preview: 'linear-gradient(135deg, #09203f 0%, #537895 100%)' },
    { id: 'jp_sakura', label: 'Sakura', category: 'japanese', preview: 'linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)' },
    { id: 'pastel_milk', label: 'Milk Pastel', category: 'pastel', preview: 'linear-gradient(135deg, #fff1eb 0%, #ace0f9 100%)' },
    { id: 'pastel_lilac', label: 'Lilac Mist', category: 'pastel', preview: 'linear-gradient(135deg, #c471f5 0%, #fa71cd 100%)' },
    { id: 'pastel_cloud', label: 'Cloud Blue', category: 'pastel', preview: 'linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)' },
    { id: 'film_gold', label: 'Golden Hour', category: 'film', preview: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)' },
    { id: 'film_retro', label: 'Retro Fade', category: 'film', preview: 'linear-gradient(135deg, #243949 0%, #517fa4 100%)' },
    { id: 'bw_soft', label: 'Soft Mono', category: 'bw', preview: 'linear-gradient(135deg, #c9c9c9 0%, #f2f2f2 100%)' },
    { id: 'bw_noir', label: 'Film Noir', category: 'bw', preview: 'linear-gradient(135deg, #000000 0%, #434343 100%)' },
];

// --- EFFECT DATA DEFINITIONS ---
interface EffectPresetOption {
    id: string;
    label: string;
    gradient: string;
    desc: string;
}

const BOKEH_PRESETS: EffectPresetOption[] = [
    { id: 'soft_pink', label: 'Soft Pink', gradient: 'radial-gradient(circle at 30% 30%, rgba(255,186,217,0.8), rgba(255,255,255,0))', desc: 'Ánh sáng pastel dịu' },
    { id: 'golden_orbit', label: 'Golden Orbit', gradient: 'radial-gradient(circle at 60% 20%, rgba(255,221,158,0.8), rgba(255,255,255,0))', desc: 'Bokeh vàng ấm' },
    { id: 'neon_pearl', label: 'Neon Pearl', gradient: 'radial-gradient(circle at 40% 70%, rgba(186,212,255,0.8), rgba(255,255,255,0))', desc: 'Tông lạnh hiện đại' }
];

const LIGHT_LEAK_PRESETS: EffectPresetOption[] = [
    { id: 'warm_sun', label: 'Sunset', gradient: 'linear-gradient(120deg, rgba(255,152,0,0.65), rgba(255,255,255,0))', desc: 'Ánh cam chiều' },
    { id: 'rose_glow', label: 'Rose Glow', gradient: 'linear-gradient(100deg, rgba(255,64,129,0.6), rgba(255,255,255,0))', desc: 'Hồng retro' },
    { id: 'aqua_flash', label: 'Aqua Flash', gradient: 'linear-gradient(90deg, rgba(33,150,243,0.5), rgba(255,255,255,0))', desc: 'Ánh xanh film' }
];


const Histogram: React.FC<{ data: HistogramData | null }> = ({ data }) => {
    const paths = useMemo(() => {
        if (!data) return null;
        const { r, g, b, l } = data;
        const max = Math.max(...r, ...g, ...b) || 1;

        const createPath = (values: number[], isFill: boolean = true) => {
            let path = `M 0 40`;
            for (let i = 0; i < 256; i++) {
                const x = (i / 255) * 100;
                const y = 40 - (values[i] / max) * 38; 
                path += ` L ${x} ${y}`;
            }
            if (isFill) {
                path += ` L 100 40 Z`;
            }
            return path;
        };

        return { r: createPath(r), g: createPath(g), b: createPath(b), l: createPath(l, false) };
    }, [data]);

    return (
        <div className="bg-zinc-900 p-3 rounded-lg border border-zinc-800 mb-4">
            <div className="flex justify-between items-center mb-2">
                <span className="text-[10px] text-gray-400 font-medium uppercase tracking-wider">Biểu đồ màu</span>
                {data && (
                    <div className="flex gap-1">
                        <div className="w-2 h-2 rounded-full bg-red-500/50"></div>
                        <div className="w-2 h-2 rounded-full bg-green-500/50"></div>
                        <div className="w-2 h-2 rounded-full bg-blue-500/50"></div>
                        <div className="w-2 h-2 rounded-full bg-white/50"></div>
                    </div>
                )}
            </div>
            <div className="h-24 w-full relative flex items-end gap-[1px] opacity-90">
                {paths ? (
                    <svg viewBox="0 0 100 40" className="w-full h-full overflow-visible" preserveAspectRatio="none">
                        <path d={paths.r} fill="rgba(239, 68, 68, 0.4)" className="mix-blend-screen" />
                        <path d={paths.g} fill="rgba(34, 197, 94, 0.4)" className="mix-blend-screen" />
                        <path d={paths.b} fill="rgba(59, 130, 246, 0.4)" className="mix-blend-screen" />
                        <path d={paths.l} fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="0.5" />
                    </svg>
                ) : (
                    <div className="w-full h-full flex flex-col items-center justify-center text-zinc-700">
                         <BarChart3 size={24} strokeWidth={1} className="mb-1 opacity-50"/>
                         <span className="text-[10px]">Chưa có dữ liệu</span>
                    </div>
                )}
                <div className="absolute inset-0 border-t border-b border-zinc-700/30 flex justify-between pointer-events-none">
                    <div className="h-full w-px bg-zinc-700/30"></div>
                    <div className="h-full w-px bg-zinc-700/30"></div>
                    <div className="h-full w-px bg-zinc-700/30"></div>
                </div>
            </div>
             <div className="flex justify-between text-[9px] text-zinc-600 mt-1 font-mono">
                <span>0</span>
                <span>128</span>
                <span>255</span>
            </div>
        </div>
    )
}

const AspectRatioBtn: React.FC<{ label: string, ratio: string, active: boolean, onClick: () => void }> = ({ label, ratio, active, onClick }) => (
    <button 
        onClick={onClick}
        className={`py-3 px-1 text-xs font-medium rounded-lg transition-all flex items-center justify-center 
            ${active 
                ? 'bg-fuchsia-500 text-white shadow-[0_0_10px_rgba(217,70,239,0.3)]' 
                : 'bg-zinc-800/50 text-gray-400 border border-zinc-700/50 hover:bg-zinc-800 hover:text-gray-200 hover:border-zinc-600'
            }`}
    >
        <span>{label}</span>
    </button>
);

const AccordionSection: React.FC<{
    title: string;
    isOpen: boolean;
    onToggle: () => void;
    children: React.ReactNode;
}> = ({ title, isOpen, onToggle, children }) => (
    <div className="border-b border-zinc-800 last:border-0">
        <button 
            onClick={onToggle}
            className="w-full flex items-center justify-between py-4 hover:bg-zinc-800/30 transition-colors px-1"
        >
            <span className="font-medium text-sm text-gray-200">{title}</span>
            {isOpen ? (
                <ChevronDown size={16} className="text-gray-500" />
            ) : (
                <Play size={10} className="text-gray-500 mr-1" fill="currentColor" />
            )}
        </button>
        {isOpen && (
            <div className="pb-6 px-1 animate-in slide-in-from-top-2 duration-200">
                {children}
            </div>
        )}
    </div>
);

const ColorOptionButton: React.FC<{
    label: string;
    colorCode: string;
    isActive: boolean;
    onClick: () => void;
    textColor?: string;
    subLabel?: string;
}> = ({ label, colorCode, isActive, onClick, textColor = "text-white", subLabel }) => (
    <button
        onClick={onClick}
        className={`h-10 rounded-lg text-[10px] font-semibold transition-all shadow-sm border flex flex-col items-center justify-center gap-0.5
            ${isActive 
                ? 'ring-2 ring-offset-2 ring-offset-zinc-900 ring-fuchsia-500 border-transparent' 
                : 'border-transparent opacity-80 hover:opacity-100 hover:scale-105'
            } ${textColor}`}
        style={{ backgroundColor: colorCode }}
    >
        <span>{label}</span>
        {subLabel && <span className="text-[8px] font-normal opacity-80">{subLabel}</span>}
    </button>
);

const LensSwatch: React.FC<{
    color: string;
    label: string;
    isActive: boolean;
    onClick: () => void;
    isEmpty?: boolean;
}> = ({ color, label, isActive, onClick, isEmpty }) => (
    <div className="flex flex-col items-center gap-1.5 group cursor-pointer" onClick={onClick}>
        <div 
            className={`w-8 h-8 rounded-full transition-all border relative overflow-hidden
                ${isActive 
                    ? 'ring-2 ring-offset-2 ring-offset-zinc-900 ring-blue-500 border-transparent scale-110' 
                    : 'border-zinc-600 group-hover:border-zinc-400'
                }`}
            style={{ backgroundColor: isEmpty ? 'transparent' : color }}
        >
            {isEmpty && (
                <div className="absolute inset-0 border-2 border-zinc-600 rounded-full" />
            )}
        </div>
        <span className={`text-[9px] text-center whitespace-nowrap ${isActive ? 'text-white' : 'text-zinc-500'}`}>{label}</span>
    </div>
);

const TogglePill: React.FC<{ label: string; active: boolean; onClick: () => void; description?: string }> = ({ label, active, onClick, description }) => (
    <button
        onClick={onClick}
        className={`w-full flex items-center justify-between px-4 py-3 rounded-2xl border text-sm transition-all
            ${active ? 'bg-fuchsia-600/10 border-fuchsia-500/40 text-white shadow-[0_0_20px_rgba(217,70,239,0.3)]' : 'bg-transparent border-zinc-800 text-gray-300 hover:border-zinc-600'}
        `}
    >
        <div className="flex flex-col text-left">
            <span className="font-semibold">{label}</span>
            {description && <span className="text-[11px] text-gray-500">{description}</span>}
        </div>
        <div className={`w-10 h-5 rounded-full transition-all ${active ? 'bg-fuchsia-500' : 'bg-zinc-700'}`}>
            <div className={`w-5 h-5 bg-white rounded-full transition-all transform ${active ? 'translate-x-5' : 'translate-x-0'}`}></div>
        </div>
    </button>
);

const FilterPresetCard: React.FC<{ preset: FilterPreset; isActive: boolean; onSelect: () => void; intensity?: number }> = ({ preset, isActive, onSelect, intensity = 1 }) => (
    <button
        onClick={onSelect}
        className={`relative rounded-2xl p-3 border transition-all flex flex-col items-start justify-between min-h-[120px]
            ${isActive ? 'border-fuchsia-500 shadow-lg shadow-fuchsia-900/30' : 'border-transparent bg-zinc-900/40 hover:border-zinc-700'}
        `}
        style={{ backgroundImage: preset.preview, opacity: 0.9 + intensity * 0.001 }}
    >
        <span className="text-xs font-semibold drop-shadow-lg">{preset.label}</span>
        <span className="text-[10px] uppercase tracking-wide mt-auto bg-black/40 px-2 py-0.5 rounded-full">{preset.category}</span>
        {isActive && (
            <div className="absolute top-2 right-2 w-6 h-6 rounded-full bg-fuchsia-600 flex items-center justify-center text-white text-[10px] font-bold">
                ✓
            </div>
        )}
    </button>
);

type AIStatus = 'idle' | 'running' | 'done' | 'error';
type PreviewEntry = { preview: string; mask?: string | null; meta?: AIPreviewMeta | null };
type AIPreviewMap = Record<string, PreviewEntry>;

type AIModule = {
    id: string;
    title: string;
    description: string;
    requiresReference?: boolean;
};

type AISection = {
    id: string;
    title: string;
    subtitle?: string;
    modules: AIModule[];
};

const AI_STATUS_LABELS: Record<AIStatus, string> = {
    idle: 'Sẵn sàng',
    running: 'Đang xử lý…',
    done: 'Hoàn tất',
    error: 'Lỗi, thử lại'
};

const AI_STATUS_CLASSES: Record<AIStatus, string> = {
    idle: 'text-emerald-400/80',
    running: 'text-amber-400',
    done: 'text-sky-400',
    error: 'text-red-400'
};

const PREVIEW_VARIANT_STYLES: Record<
    string,
    { border: string; chipBg: string; chipText: string; accentText: string }
> = {
    people: {
        border: 'border-rose-500/40',
        chipBg: 'bg-rose-500/15',
        chipText: 'text-rose-200',
        accentText: 'text-rose-200'
    },
    object: {
        border: 'border-emerald-500/40',
        chipBg: 'bg-emerald-500/15',
        chipText: 'text-emerald-200',
        accentText: 'text-emerald-200'
    },
    remove_bg: {
        border: 'border-cyan-500/40',
        chipBg: 'bg-cyan-500/15',
        chipText: 'text-cyan-200',
        accentText: 'text-cyan-200'
    },
    quality_superres: {
        border: 'border-emerald-500/40',
        chipBg: 'bg-emerald-500/15',
        chipText: 'text-emerald-100',
        accentText: 'text-emerald-200'
    },
    quality_restore: {
        border: 'border-amber-500/40',
        chipBg: 'bg-amber-500/15',
        chipText: 'text-amber-100',
        accentText: 'text-amber-200'
    },
    color_clone: {
        border: 'border-cyan-500/40',
        chipBg: 'bg-cyan-500/15',
        chipText: 'text-cyan-100',
        accentText: 'text-cyan-200'
    },
    generic: {
        border: 'border-zinc-800/60',
        chipBg: 'bg-zinc-800/40',
        chipText: 'text-gray-300',
        accentText: 'text-gray-300'
    }
};

const PREVIEW_VARIANT_COPY: Record<string, string> = {
    remove_bg: 'PNG nền trong suốt với chủ thể giữ nguyên màu sắc, sẵn sàng ghép background mới.',
    people: 'PNG chân dung đã mịn viền tóc và vai, dễ ghép nền hoặc chèn background AI.',
    object: 'PNG sticker có bóng đổ và viền neon, sẵn sàng dán vào poster/catalog.',
    quality_superres: 'Ảnh đã upscale 4K, chi tiết rõ hơn và nhiễu được giảm tối ưu.',
    quality_restore: 'Ảnh đen trắng đã được tô màu tự động, tái tạo màu sắc chân thực và sống động.',
    color_clone: 'Ảnh đã mang bảng màu của ảnh tham chiếu, giữ nguyên chi tiết.',
    generic: 'Ảnh xem trước từ mô-đun AI để bạn áp dụng hoặc tải về.'
};

const AI_SECTIONS: AISection[] = [
    {
        id: 'ai_beauty',
        title: 'I. AI Tự động làm đẹp',
        subtitle: 'Sức mạnh AI tự động vượt trội',
        modules: [
            {
                id: 'ai_beauty_full',
                title: 'Tự chỉnh toàn diện',
                description: 'AI phân tích và tự động cân bằng ánh sáng, màu sắc, độ nét cho toàn bộ ảnh. Phù hợp mọi loại ảnh.'
            },
            {
                id: 'ai_beauty_portrait',
                title: 'Tối ưu chân dung',
                description: 'Tập trung làm đẹp da, sáng mắt, cân bằng ánh sáng khuôn mặt. Chuyên biệt cho ảnh chân dung.'
            },
            {
                id: 'ai_beauty_tone',
                title: 'AI Smart Tone',
                description: 'Phân tích ảnh và gợi ý tone màu phù hợp (warm/cool/cinematic/vintage...). Tạo phong cách màu sắc chuyên nghiệp.'
            }
        ]
    },
    {
        id: 'ai_quality',
        title: 'II. Nâng cấp chất lượng',
        modules: [
            {
                id: 'ai_quality_superres',
                title: 'Siêu phân giải 4K',
                description: 'ESRGAN phục hồi chi tiết, giảm nhiễu mạnh'
            },
            {
                id: 'ai_quality_restore',
                title: 'Phục hồi ảnh đen trắng',
                description: 'Tô màu tự động cho ảnh đen trắng, tái tạo màu sắc chân thực'
            }
        ]
    },
    {
        id: 'ai_cutout',
        title: 'III. Xóa Background',
        modules: [
            {
                id: 'ai_cutout_remove',
                title: 'Xóa Background',
                description: 'Tách nền tự động, trả PNG trong suốt giữ nguyên màu sắc'
            }
        ]
    },
    {
        id: 'ai_color',
        title: 'IV. AI Chuyển màu & Style',
        modules: [
            {
                id: 'ai_color_transfer',
                title: 'Chuyển màu theo ảnh tham chiếu',
                description: 'Sao chép tone màu và ánh sáng từ ảnh khác',
                requiresReference: true
            }
        ]
    },
    {
        id: 'ai_scene',
        title: 'V. AI Nâng cao',
        subtitle: 'Ghép mặt với outfit & bối cảnh mới',
        modules: [
            {
                id: 'ai_scene_gen',
                title: 'Ghép mặt & Bối cảnh',
                description: 'Gemini 2.5 vẽ lại khuôn mặt hiện tại trong trang phục và địa điểm mới'
            }
        ]
    }
];

const EffectCard: React.FC<{ title: string; description: string; icon: React.ElementType; children: React.ReactNode }> = ({ title, description, icon: Icon, children }) => (
    <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-4 space-y-3">
        <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-zinc-800 flex items-center justify-center">
                <Icon size={18} className="text-fuchsia-400" />
            </div>
            <div>
                <p className="text-sm font-semibold text-gray-100">{title}</p>
                <p className="text-xs text-gray-500">{description}</p>
            </div>
        </div>
        {children}
    </div>
);

export const Sidebar: React.FC<SidebarProps> = ({ 
    activeTab, histogramData, basicValues, onBasicChange, onAutoAdjust, 
    transformValues, onTransformChange, effectsValues, onEffectsChange,
    textLayers, activeTextId, onAddText, onUpdateText,
    beautyValues, onBeautyChange, onApplyBeautyPreset,
    isManualAcneMode, toggleManualAcneMode,
    filterValues, onFilterChange,
    onAIProAction,
    aiProStatus,
    aiProErrors,
    aiProInsights,
    aiProPreviews,
    onApplyAiPreview,
    onDismissAiPreview,
    currentImage,
    onAIImageGenerated
}) => {
  
  const updateBasicValue = (key: string, val: number) => {
    onBasicChange((prev: any) => ({ ...prev, [key]: val }));
  };

  const activeTextLayer = useMemo(() => {
      return textLayers?.find(l => l.id === activeTextId) || null;
  }, [textLayers, activeTextId]);

  const updateText = (key: keyof TextLayer, val: any) => {
      if (activeTextId && onUpdateText) {
          onUpdateText(activeTextId, { [key]: val });
      }
  };

  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
      light: true, color: true, detail: true,
      skin: true, face: true, eyes: false, mouth: false, hair: false,
      tools: true, crop: true,
      eff_light: true, eff_sparkle: false, eff_art: false, eff_vintage: false, eff_depth: false, eff_focus: false, eff_special: false
  });
  const [aiSectionsOpen, setAiSectionsOpen] = useState<Record<string, boolean>>(() => {
      const initial: Record<string, boolean> = {};
      AI_SECTIONS.forEach((section, index) => {
          initial[section.id] = index === 0;
      });
      return initial;
  });
  const [selectedAIModule, setSelectedAIModule] = useState<string | null>(AI_SECTIONS[0]?.modules[0]?.id ?? null);
  const [aiIntensity, setAiIntensity] = useState<number>(75);
  const [isAIGenerating, setIsAIGenerating] = useState(false);
  const [sceneInputs, setSceneInputs] = useState<{ outfit: string; location: string }>({ outfit: '', location: '' });
  const [sceneError, setSceneError] = useState<string | null>(null);
  const [referenceImageFile, setReferenceImageFile] = useState<File | null>(null);
  const [referenceImagePreview, setReferenceImagePreview] = useState<string | null>(null);
  const geminiSceneModel = DEFAULT_GEMINI_SCENE_MODEL;

  const toggleSection = (section: string) => {
      setExpandedSections(prev => ({...prev, [section]: !prev[section]}));
  };

  const toggleAiSection = (sectionId: string) => {
      setAiSectionsOpen(prev => ({ ...prev, [sectionId]: !prev[sectionId] }));
  };

  const selectedModuleData = useMemo(() => {
      if (!selectedAIModule) return null;
      for (const section of AI_SECTIONS) {
          const mod = section.modules.find(module => module.id === selectedAIModule);
          if (mod) return mod;
      }
      return null;
  }, [selectedAIModule]);
  const isSceneGenerationModule = selectedAIModule === 'ai_scene_gen';
  const requiresReferenceImage = selectedModuleData?.requiresReference ?? false;
  const baseModuleStatus: AIStatus = selectedAIModule ? (aiProStatus?.[selectedAIModule] ?? 'idle') : 'idle';
  const selectedModuleStatus: AIStatus = isSceneGenerationModule && isAIGenerating ? 'running' : baseModuleStatus;
  const activeAIInsight = selectedAIModule ? aiProInsights?.[selectedAIModule] : null;
  const selectedPreview = selectedAIModule ? aiProPreviews?.[selectedAIModule] : null;
    const previewMeta = selectedPreview?.meta ?? null;
    const previewVariant = previewMeta?.variant ?? 'generic';
    const previewStyle = PREVIEW_VARIANT_STYLES[previewVariant] ?? PREVIEW_VARIANT_STYLES.generic;
    const previewDescription = PREVIEW_VARIANT_COPY[previewVariant] ?? PREVIEW_VARIANT_COPY.generic;
    const previewDownloadLabel =
        previewVariant === 'remove_bg' || previewVariant === 'people' || previewVariant === 'object'
            ? 'Tải PNG trong suốt'
            : 'Tải ảnh nâng cấp';
    const activeAIMetricLabel = useMemo(() => {
      if (!activeAIInsight?.metrics) return '';
      const metrics = activeAIInsight.metrics as Record<string, unknown>;
      // For AI Smart Tone, show the recommended tone name
      if (metrics.toneName) {
          return `Tone: ${metrics.toneName}`;
      }
      // For other modules
      const candidate = metrics.expectedGain ?? metrics.expected_gain ?? metrics.primaryModel;
      return typeof candidate === 'string' ? candidate : '';
  }, [activeAIInsight]);

  useEffect(() => {
      if (!selectedAIModule) return;
      const status = aiProStatus?.[selectedAIModule];
      if (status && status !== 'running') {
          setIsAIGenerating(false);
      }
  }, [aiProStatus, selectedAIModule]);

  useEffect(() => {
      if (!isSceneGenerationModule && sceneError) {
          setSceneError(null);
      }
  }, [isSceneGenerationModule, sceneError]);

  useEffect(() => {
      if (sceneError) {
          setSceneError(null);
      }
  }, [sceneInputs]);

  useEffect(() => {
      if (!requiresReferenceImage) {
          setReferenceImageFile(null);
          setReferenceImagePreview(null);
      }
  }, [requiresReferenceImage]);

  const handleReferenceImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
          setReferenceImageFile(file);
          const reader = new FileReader();
          reader.onloadend = () => {
              setReferenceImagePreview(reader.result as string);
          };
          reader.readAsDataURL(file);
      }
  };

  const updateBeautyValue = (category: keyof BeautyValues, key: string, val: number) => {
      if (!onBeautyChange || !beautyValues) return;
      onBeautyChange(prev => ({ ...prev, [category]: { ...(prev[category] as object), [key]: val } }));
  };


  const visibleFilters = useMemo(() => {
      return FILTERS_DATA.filter(f => f.category === filterValues.selectedCategory);
  }, [filterValues.selectedCategory]);

  const convertSourceToBase64 = async (source: string): Promise<{ base64: string; mimeType: string }> => {
      if (source.startsWith('data:')) {
          const [header, data] = source.split(',');
          const mimeMatch = header.match(/data:(.*?);base64/);
          const mimeType = mimeMatch?.[1] ?? 'image/jpeg';
          return { base64: data ?? '', mimeType };
      }
      const response = await fetch(source);
      if (!response.ok) {
          throw new Error('Không thể tải ảnh tham chiếu.');
      }
      const blob = await response.blob();
      const reader = new FileReader();
      const converted = await new Promise<string>((resolve, reject) => {
          reader.onerror = () => reject(new Error('Không thể đọc ảnh tham chiếu.'));
          reader.onloadend = () => resolve((reader.result as string) || '');
          reader.readAsDataURL(blob);
      });
      const [header, data] = converted.split(',');
      const mimeMatch = header.match(/data:(.*?);base64/);
        const mimeType = (mimeMatch?.[1] ?? blob.type) || 'image/png';
      return { base64: data ?? '', mimeType };
  };

const buildScenePrompt = (outfit: string, location: string) =>
    `Create a high-quality, photorealistic image of a person wearing ${outfit} standing at ${location}. The person should have the facial structure and features of the person in the provided reference image. The lighting should be cinematic, 8k resolution, highly detailed texture, soft bokeh background. Preserve the identity of the reference face.`;

const toHttpStatus = (value: unknown): number | null => {
    if (typeof value === 'number' && Number.isFinite(value)) {
        return value;
    }
    if (typeof value === 'string' && /^\d+$/.test(value)) {
        return Number(value);
    }
    return null;
};

const getGeminiErrorStatus = (error: unknown): number | null => {
    if (!error || typeof error !== 'object') {
        return null;
    }
    const errorObj = error as Record<string, unknown>;
    return (
        toHttpStatus(errorObj.status) ??
        toHttpStatus(errorObj.statusCode) ??
        toHttpStatus(errorObj.code)
    );
};

const deriveSceneErrorMessage = (error: unknown) => {
    const fallback = 'Gemini gặp sự cố khi tạo ảnh. Vui lòng thử lại.';
    if (!error) {
        return fallback;
    }
    const status = getGeminiErrorStatus(error);
    const message =
        error instanceof Error
            ? error.message
            : typeof error === 'string'
              ? error
              : '';
    const normalized = message.toLowerCase();
    if (status === 404 || normalized.includes('404')) {
        return 'Gemini trả về 404 (Model không tồn tại hoặc không hỗ trợ ảnh). Hãy dùng gemini-2.5-flash-image và kiểm tra biến môi trường.';
    }
    if (status === 429 || normalized.includes('429')) {
        return 'Gemini trả về 429 (Hết hạn ngạch). Nâng quota dự án hoặc thử lại sau ít phút.';
    }
    return message || fallback;
};

  const runSceneGeneration = async () => {
      if (!currentImage) {
          setSceneError('Vui lòng tải ảnh lên Canvas trước.');
          return;
      }
      
      const outfit = sceneInputs.outfit.trim();
      const location = sceneInputs.location.trim();
      if (!outfit || !location) {
          setSceneError('Nhập đầy đủ “Trang phục” và “Địa điểm”.');
          return;
      }
      try {
          setSceneError(null);
          setIsAIGenerating(true);
          const { base64, mimeType } = await convertSourceToBase64(currentImage);
          const prompt = buildScenePrompt(outfit, location);
          const { dataUrl } = await generateSceneImage({
              model: geminiSceneModel,
              prompt,
              base64Image: base64,
              mimeType
          });
          onAIImageGenerated?.(dataUrl);
      } catch (error) {
          console.error('Gemini scene generation lỗi:', error);
          setSceneError(deriveSceneErrorMessage(error));
      } finally {
          setIsAIGenerating(false);
      }
  };

  const resetBasic = () => {
       onBasicChange({
        exposure: 0, brightness: 0, contrast: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0,
        temp: 0, tint: 0, vibrance: 0, saturation: 0, hue: 0, grayscale: 0,
        sharpen: 0, blur: 0, clarity: 0, texture: 0, dehaze: 0, denoise: 0
      });
  }

  // Transform Helpers
  const rotate = (deg: number) => {
      const current = transformValues.rotate;
      onTransformChange({ ...transformValues, rotate: current + deg });
  };

  const flip = (direction: 'horizontal' | 'vertical') => {
      if (direction === 'horizontal') {
          onTransformChange({ ...transformValues, flipHorizontal: !transformValues.flipHorizontal });
      } else {
           onTransformChange({ ...transformValues, flipVertical: !transformValues.flipVertical });
      }
  };
  
  const setRatio = (ratio: string) => {
      // When changing aspect ratio, automatically create crop with that ratio
      if (ratio === 'original' || ratio === 'free') {
          onTransformChange({ ...transformValues, aspectRatio: ratio, crop: undefined });
      } else {
          // Calculate crop dimensions based on aspect ratio
          const parts = ratio.split(':');
          if (parts.length === 2) {
              const targetRatio = parseFloat(parts[0]) / parseFloat(parts[1]);
              if (!isNaN(targetRatio) && targetRatio > 0) {
                  // Create centered crop with the target ratio
                  // Assume image is 100x100 for calculation, then scale
                  let cropW = 80; // Default 80% width
                  let cropH = cropW / targetRatio;
                  
                  // If height exceeds bounds, adjust
                  if (cropH > 80) {
                      cropH = 80;
                      cropW = cropH * targetRatio;
                  }
                  
                  const cropX = (100 - cropW) / 2;
                  const cropY = (100 - cropH) / 2;
                  
                  onTransformChange({ 
                      ...transformValues, 
                      aspectRatio: ratio,
                      crop: { x: cropX, y: cropY, width: cropW, height: cropH }
                  });
              }
          }
      }
  };

  const resetCrop = () => {
      onTransformChange({ ...transformValues, crop: undefined });
  };

  const resetTransform = () => {
      onTransformChange({
          ...transformValues,
          rotate: 0,
          rotateFree: 0,
          flipHorizontal: false,
          flipVertical: false,
          straighten: 0,
          aspectRatio: 'original'
      });
  };

  const handleSelectAIModule = (moduleId: string) => {
      setSelectedAIModule(moduleId);
  };

  const handleAIGenerate = () => {
      if (!selectedAIModule || isAIGenerating) return;
      if (isSceneGenerationModule) {
          runSceneGeneration();
          return;
      }
      setIsAIGenerating(true);
      const maybePromise = onAIProAction?.(selectedAIModule, { 
          intensity: aiIntensity,
          referenceImageFile: requiresReferenceImage ? referenceImageFile : null,
      });
      if (maybePromise && typeof (maybePromise as Promise<void>).finally === 'function') {
          (maybePromise as Promise<void>).finally(() => setIsAIGenerating(false));
      } else if (!aiProStatus) {
          setTimeout(() => setIsAIGenerating(false), 800);
      }
  };

  const missingSceneInputs = isSceneGenerationModule && (!sceneInputs.outfit.trim() || !sceneInputs.location.trim());
  const missingSceneSource = isSceneGenerationModule && !currentImage;
  const missingReference = requiresReferenceImage && !referenceImageFile;
  const isGenerateDisabled =
      !selectedAIModule ||
      isAIGenerating ||
      selectedModuleStatus === 'running' ||
      missingSceneInputs ||
      missingSceneSource ||
      missingReference;

  const FONTS = [
      { name: 'Inter', label: 'Mặc định' },
      { name: 'Times New Roman', label: 'Serif' },
      { name: 'Courier New', label: 'Mono' },
      { name: 'Cursive', label: 'Viết tay' },
      { name: 'Impact', label: 'Dày' }
  ];

  return (
    <aside className="w-[320px] bg-[#18181b] border-l border-zinc-800 flex flex-col h-full shrink-0 z-10 font-sans">
        <div className="h-14 px-6 border-b border-zinc-800 flex flex-col justify-center shrink-0">
            <h2 className="text-gray-100 font-semibold text-sm">
                {activeTab === TabType.BASIC && "Điều chỉnh cơ bản"}
                {activeTab === TabType.CROP && "CẮT & XOAY"}
                {activeTab === TabType.TEXT && "Thêm văn bản"}
                {activeTab === TabType.BEAUTY && "Làm đẹp chuyên sâu"}
                {activeTab === TabType.FILTER && "Bộ lọc & Màu"}
                {activeTab === TabType.EFFECT && "Hiệu ứng đặc biệt"}
                {activeTab === TabType.AI_PRO && "AI PRO Nâng cao"}
            </h2>
            <p className="text-xs text-gray-500 mt-0.5">
                {activeTab === TabType.CROP ? "Thay đổi kích thước khung hình" : 
                 activeTab === TabType.FILTER ? "Khám phá các bộ lọc màu theo phong cách riêng" :
                 activeTab === TabType.EFFECT ? "Thêm hiệu ứng ánh sáng, nghệ thuật" :
                 activeTab === TabType.TEXT ? "Thêm và chỉnh sửa văn bản trên ảnh" :
                 activeTab === TabType.BEAUTY ? "Chỉnh da, mặt và trang điểm" :
                 activeTab === TabType.AI_PRO ? "Kích hoạt các mô-đun AI chất lượng cao theo yêu cầu" :
                 "Tinh chỉnh từng chi tiết ảnh của bạn."}
            </p>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4 custom-scrollbar">
            {(activeTab === TabType.BASIC || activeTab === TabType.FILTER) && <Histogram data={histogramData} />}

            {/* === BASIC TAB === */}
            {activeTab === TabType.BASIC && (
                <div className="space-y-1">
                    <div className="mb-4">
                        <button 
                            onClick={onAutoAdjust}
                            className="w-full py-2 bg-gradient-to-r from-fuchsia-600 to-purple-600 rounded-lg text-white flex items-center justify-center gap-2 shadow-lg shadow-purple-900/20 hover:shadow-purple-900/40 transition-all hover:scale-[1.02]"
                        >
                            <Wand2 size={16} />
                            <span className="font-medium text-xs">Tự động điều chỉnh AI</span>
                        </button>
                    </div>

                    <AccordionSection title="Ánh sáng (Light)" isOpen={expandedSections.light} onToggle={() => toggleSection('light')}>
                         <SliderControl label="Độ sáng (Exposure)" value={basicValues.exposure} min={-100} max={100} onChange={(v) => updateBasicValue('exposure', v)} />
                         <SliderControl label="Độ tương phản (Contrast)" value={basicValues.contrast} min={-100} max={100} onChange={(v) => updateBasicValue('contrast', v)} />
                         <SliderControl label="Vùng sáng (Highlights)" value={basicValues.highlights} min={-100} max={100} onChange={(v) => updateBasicValue('highlights', v)} />
                         <SliderControl label="Vùng tối (Shadows)" value={basicValues.shadows} min={-100} max={100} onChange={(v) => updateBasicValue('shadows', v)} />
                         <SliderControl label="Điểm trắng (Whites)" value={basicValues.whites} min={-100} max={100} onChange={(v) => updateBasicValue('whites', v)} />
                         <SliderControl label="Điểm đen (Blacks)" value={basicValues.blacks} min={-100} max={100} onChange={(v) => updateBasicValue('blacks', v)} />
                    </AccordionSection>

                    <AccordionSection title="Màu sắc (Color)" isOpen={expandedSections.color} onToggle={() => toggleSection('color')}>
                         <SliderControl label="Nhiệt độ (Temp)" value={basicValues.temp} min={-100} max={100} onChange={(v) => updateBasicValue('temp', v)} />
                         <SliderControl label="Sắc thái (Tint)" value={basicValues.tint} min={-100} max={100} onChange={(v) => updateBasicValue('tint', v)} />
                         <SliderControl label="Độ bão hòa (Saturation)" value={basicValues.saturation} min={-100} max={100} onChange={(v) => updateBasicValue('saturation', v)} />
                         <SliderControl label="Độ rực rỡ (Vibrance)" value={basicValues.vibrance} min={-100} max={100} onChange={(v) => updateBasicValue('vibrance', v)} />
                         <SliderControl label="Đổi màu (Hue)" value={basicValues.hue} min={-180} max={180} onChange={(v) => updateBasicValue('hue', v)} />
                         <SliderControl label="Đen trắng (B&W)" value={basicValues.grayscale} min={0} max={100} onChange={(v) => updateBasicValue('grayscale', v)} />
                    </AccordionSection>

                    <AccordionSection title="Chi tiết (Detail)" isOpen={expandedSections.detail} onToggle={() => toggleSection('detail')}>
                         <SliderControl label="Độ rõ nét (Clarity)" value={basicValues.clarity} min={-100} max={100} onChange={(v) => updateBasicValue('clarity', v)} />
                         <SliderControl label="Sắc nét (Sharpen)" value={basicValues.sharpen} min={0} max={100} onChange={(v) => updateBasicValue('sharpen', v)} />
                         <SliderControl label="Giảm nhiễu (Denoise)" value={basicValues.denoise} min={0} max={100} onChange={(v) => updateBasicValue('denoise', v)} />
                         <SliderControl label="Khử sương (Dehaze)" value={basicValues.dehaze} min={-100} max={100} onChange={(v) => updateBasicValue('dehaze', v)} />
                         <SliderControl label="Hạt (Texture)" value={basicValues.texture} min={0} max={100} onChange={(v) => updateBasicValue('texture', v)} />
                         <SliderControl label="Làm mờ (Blur)" value={basicValues.blur} min={0} max={100} onChange={(v) => updateBasicValue('blur', v)} />
                    </AccordionSection>

                    <div className="pt-4">
                         <button onClick={resetBasic} className="text-xs text-red-400 hover:text-red-300 w-full text-center py-2">Đặt lại tất cả</button>
                    </div>
                </div>
            )}
            
            {/* === CROP TAB === */}
            {activeTab === TabType.CROP && (
                <div className="space-y-8 mt-2">

                    {/* Aspect Ratio Section */}
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <h3 className="text-xs font-semibold text-gray-300 uppercase tracking-wide opacity-80">TỈ LỆ KHUNG ẢNH</h3>
                        </div>
                        <div className="grid grid-cols-3 gap-3">
                            <AspectRatioBtn label="Gốc" ratio="original" active={transformValues.aspectRatio === 'original'} onClick={() => setRatio('original')} />
                            <AspectRatioBtn label="Tự do" ratio="free" active={transformValues.aspectRatio === 'free'} onClick={() => setRatio('free')} />
                            <AspectRatioBtn label="1:1" ratio="1:1" active={transformValues.aspectRatio === '1:1'} onClick={() => setRatio('1:1')} />
                            <AspectRatioBtn label="3:4" ratio="3:4" active={transformValues.aspectRatio === '3:4'} onClick={() => setRatio('3:4')} />
                            <AspectRatioBtn label="4:5" ratio="4:5" active={transformValues.aspectRatio === '4:5'} onClick={() => setRatio('4:5')} />
                            <AspectRatioBtn label="16:9" ratio="16:9" active={transformValues.aspectRatio === '16:9'} onClick={() => setRatio('16:9')} />
                            <AspectRatioBtn label="9:16" ratio="9:16" active={transformValues.aspectRatio === '9:16'} onClick={() => setRatio('9:16')} />
                        </div>
                        <p className="text-xs text-gray-500 mt-2">Chọn tỉ lệ để tự động cắt ảnh</p>
                        <button 
                            onClick={resetCrop}
                            className="w-full mt-3 py-2.5 px-4 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-lg text-xs font-medium text-gray-300 hover:text-white transition-all flex items-center justify-center gap-2"
                        >
                            <RotateCcw size={14} />
                            Đặt lại vùng cắt
                        </button>
                    </div>

                    {/* Rotate & Flip Section */}
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <h3 className="text-sm font-semibold text-gray-200">Xoay & Lật</h3>
                                <ChevronDown size={14} className="text-gray-500" />
                            </div>
                            <button 
                                onClick={resetTransform}
                                className="text-xs text-fuchsia-500 hover:text-fuchsia-400 px-2 py-1 rounded bg-fuchsia-500/10 border border-fuchsia-500/20"
                            >
                                Đặt lại
                            </button>
                        </div>

                        {/* Action Buttons */}
                        <div className="grid grid-cols-4 gap-3">
                             <button onClick={() => rotate(-90)} className="aspect-square rounded-xl bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-gray-400 hover:text-gray-200 flex flex-col items-center justify-center transition-all group">
                                <RotateCcw size={20} className="mb-1 group-hover:-rotate-90 transition-transform duration-300" />
                                <span className="text-[10px]">-90°</span>
                             </button>
                             <button onClick={() => rotate(90)} className="aspect-square rounded-xl bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-gray-400 hover:text-gray-200 flex flex-col items-center justify-center transition-all group">
                                <RotateCw size={20} className="mb-1 group-hover:rotate-90 transition-transform duration-300" />
                                <span className="text-[10px]">+90°</span>
                             </button>
                             <button onClick={() => flip('horizontal')} className={`aspect-square rounded-xl border flex items-center justify-center transition-all ${transformValues.flipHorizontal ? 'bg-fuchsia-600 text-white border-fuchsia-500 shadow-lg shadow-fuchsia-500/20' : 'bg-zinc-800 text-gray-400 border-zinc-700 hover:bg-zinc-700 hover:text-gray-200'}`}>
                                <FlipHorizontal size={22} />
                             </button>
                             <button onClick={() => flip('vertical')} className={`aspect-square rounded-xl border flex items-center justify-center transition-all ${transformValues.flipVertical ? 'bg-fuchsia-600 text-white border-fuchsia-500 shadow-lg shadow-fuchsia-500/20' : 'bg-zinc-800 text-gray-400 border-zinc-700 hover:bg-zinc-700 hover:text-gray-200'}`}>
                                <FlipVertical size={22} />
                             </button>
                        </div>

                        {/* Sliders */}
                        <div className="pt-4 space-y-6">
                             <SliderControl 
                                label="Xoay tự do" 
                                value={transformValues.rotateFree} 
                                min={-180} max={180} 
                                onChange={(v) => onTransformChange({...transformValues, rotateFree: v})} 
                             />
                             <SliderControl 
                                label="Căn thẳng" 
                                value={transformValues.straighten} 
                                min={-45} max={45} 
                                onChange={(v) => onTransformChange({...transformValues, straighten: v})} 
                             />
                        </div>
                    </div>
                </div>
            )}

             {/* === TEXT TAB === */}
            {activeTab === TabType.TEXT && (
                <div className="space-y-6 mt-2">
                    <button 
                        onClick={onAddText}
                        className="w-full py-3 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-lg text-white flex items-center justify-center gap-2 transition-all hover:shadow-md"
                    >
                        <Type size={18} className="text-fuchsia-500" />
                        <span className="font-medium text-sm">Thêm văn bản</span>
                    </button>
                    
                    <div className={`space-y-5 transition-opacity duration-200 ${!activeTextId ? 'opacity-50 pointer-events-none' : 'opacity-100'}`}>
                         <div className="bg-zinc-900/50 p-3 rounded-lg border border-zinc-800 space-y-2">
                             <label className="text-xs text-gray-400">Nội dung</label>
                             <textarea 
                                value={activeTextLayer?.text || ''}
                                onChange={(e) => updateText('text', e.target.value)}
                                rows={3}
                                className="w-full bg-zinc-800 border-none rounded text-sm text-gray-200 p-2 focus:ring-1 focus:ring-fuchsia-500 outline-none resize-none"
                             />
                         </div>

                         {/* Font Selection */}
                         <div className="space-y-2">
                            <label className="text-xs text-gray-400">Phông chữ</label>
                            <select 
                                value={activeTextLayer?.fontFamily || 'Inter'} 
                                onChange={(e) => updateText('fontFamily', e.target.value)}
                                className="w-full bg-zinc-800 border border-zinc-700 text-gray-200 text-xs rounded-lg p-2 focus:ring-1 focus:ring-fuchsia-500 outline-none"
                            >
                                {FONTS.map(f => <option key={f.name} value={f.name}>{f.label} ({f.name})</option>)}
                            </select>
                         </div>

                         {/* Style & Align */}
                         <div className="grid grid-cols-2 gap-3">
                             <div className="bg-zinc-800/50 p-1 rounded-lg border border-zinc-800 flex justify-between">
                                 <button onClick={() => updateText('align', 'left')} className={`p-1.5 rounded flex-1 flex justify-center ${activeTextLayer?.align === 'left' ? 'bg-zinc-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}><AlignLeft size={16}/></button>
                                 <button onClick={() => updateText('align', 'center')} className={`p-1.5 rounded flex-1 flex justify-center ${activeTextLayer?.align === 'center' ? 'bg-zinc-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}><AlignCenter size={16}/></button>
                                 <button onClick={() => updateText('align', 'right')} className={`p-1.5 rounded flex-1 flex justify-center ${activeTextLayer?.align === 'right' ? 'bg-zinc-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}><AlignRight size={16}/></button>
                             </div>
                             <div className="bg-zinc-800/50 p-1 rounded-lg border border-zinc-800 flex justify-between">
                                 <button onClick={() => updateText('isBold', !activeTextLayer?.isBold)} className={`p-1.5 rounded flex-1 flex justify-center ${activeTextLayer?.isBold ? 'bg-zinc-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}><Bold size={16}/></button>
                                 <button onClick={() => updateText('isItalic', !activeTextLayer?.isItalic)} className={`p-1.5 rounded flex-1 flex justify-center ${activeTextLayer?.isItalic ? 'bg-zinc-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}><Italic size={16}/></button>
                                 <button onClick={() => updateText('isUnderline', !activeTextLayer?.isUnderline)} className={`p-1.5 rounded flex-1 flex justify-center ${activeTextLayer?.isUnderline ? 'bg-zinc-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}><Underline size={16}/></button>
                             </div>
                         </div>

                         <SliderControl label="Cỡ chữ" value={activeTextLayer?.fontSize || 40} onChange={(v) => updateText('fontSize', v)} min={10} max={200} />
                         <SliderControl label="Độ mờ" value={activeTextLayer?.opacity || 100} onChange={(v) => updateText('opacity', v)} min={0} max={100} />
                         <SliderControl label="Giãn dòng" value={Math.round((activeTextLayer?.lineHeight || 1.2) * 10)} onChange={(v) => updateText('lineHeight', v/10)} min={8} max={30} />
                         <SliderControl label="Giãn cách chữ" value={activeTextLayer?.letterSpacing || 0} onChange={(v) => updateText('letterSpacing', v)} min={-5} max={20} />
                         
                         <div className="pt-4 border-t border-zinc-800">
                             <label className="text-xs text-gray-400 mb-2 block">Màu sắc</label>
                             <div className="grid grid-cols-6 gap-2">
                                 {['#ffffff', '#000000', '#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#a855f7', '#ec4899', '#f43f5e', '#94a3b8', '#64748b'].map(c => (
                                     <button 
                                        key={c} 
                                        onClick={() => updateText('color', c)}
                                        className={`w-8 h-8 rounded-full border transition-transform hover:scale-110 ${activeTextLayer?.color === c ? 'border-white ring-1 ring-white/50' : 'border-zinc-700'}`} 
                                        style={{backgroundColor: c}}
                                     />
                                 ))}
                             </div>
                         </div>
                    </div>
                </div>
            )}

            {/* === BEAUTY TAB === */}
            {activeTab === TabType.BEAUTY && beautyValues && onBeautyChange && (
                <div className="space-y-1">
                    <AccordionSection title="Da (Skin)" isOpen={expandedSections.skin} onToggle={() => toggleSection('skin')}>
                         {/* 1. Smoothing Modes & Slider */}
                         <div className="mb-4">
                            <div className="flex justify-between items-center mb-2">
                                <label className="text-xs font-medium text-gray-300">1. Mịn da (Smooth)</label>
                            </div>
                            
                            <div className="bg-zinc-900/50 p-2 rounded-lg border border-zinc-800 mb-3">
                                <div className="flex items-center justify-between mb-2 px-1">
                                    <span className="text-[10px] text-gray-400 uppercase">Chế độ làm mịn</span>
                                </div>
                                <div className="flex gap-1">
                                    <button 
                                        onClick={() => onBeautyChange(prev => ({...prev, skinMode: 'natural'}))}
                                        className={`flex-1 py-1.5 text-[10px] font-medium rounded-md transition-all flex items-center justify-center gap-1
                                        ${beautyValues.skinMode === 'natural' ? 'bg-fuchsia-600 text-white shadow-sm' : 'bg-zinc-800 text-gray-500 hover:text-gray-300'}`}
                                    >
                                        {beautyValues.skinMode === 'natural' && <Sparkles size={10} />}
                                        Tự nhiên
                                    </button>
                                    <button 
                                        onClick={() => onBeautyChange(prev => ({...prev, skinMode: 'strong'}))}
                                        className={`flex-1 py-1.5 text-[10px] font-medium rounded-md transition-all flex items-center justify-center gap-1
                                        ${beautyValues.skinMode === 'strong' ? 'bg-fuchsia-600 text-white shadow-sm' : 'bg-zinc-800 text-gray-500 hover:text-gray-300'}`}
                                    >
                                        {beautyValues.skinMode === 'strong' && <Sparkles size={10} fill="currentColor" />}
                                        Mạnh (Strong)
                                    </button>
                                </div>
                            </div>

                            <SliderControl 
                                label="Cường độ mịn" 
                                value={beautyValues.skinValues.smooth} 
                                onChange={(v) => updateBeautyValue('skinValues', 'smooth', v)} 
                            />
                        </div>

                        <SliderControl label="2. Trắng da (Whitening AI)" value={beautyValues.skinValues.whiten} onChange={(v) => updateBeautyValue('skinValues', 'whiten', v)} />
                        <SliderControl label="3. Đều màu da (Even Tone)" value={beautyValues.skinValues.even} onChange={(v) => updateBeautyValue('skinValues', 'even', v)} />

                        {/* Acne Removal */}
                        <div className="pt-4 border-t border-zinc-800/50 mb-4 mt-4">
                            <label className="text-xs text-gray-400 mb-3 block font-medium uppercase tracking-wide">4. Xóa mụn (Acne)</label>
                            
                            {/* Manual Only */}
                            <button 
                                onClick={toggleManualAcneMode}
                                className={`w-full py-2 rounded-lg border flex items-center justify-center gap-2 text-xs font-medium transition-all
                                    ${isManualAcneMode 
                                        ? 'bg-fuchsia-600 border-fuchsia-500 text-white shadow-md' 
                                        : 'bg-zinc-800 border-zinc-700 text-gray-400 hover:bg-zinc-700'}`}
                            >
                                <Fingerprint size={14} />
                                {isManualAcneMode ? "Đang xóa thủ công (Nhấn để tắt)" : "Chấm mụn thủ công"}
                            </button>
                        </div>

                        <SliderControl label="5. Da bóng Hàn Quốc (Dewy)" value={beautyValues.skinValues.korean} onChange={(v) => updateBeautyValue('skinValues', 'korean', v)} />
                        <SliderControl label="6. Giữ kết cấu (Texture)" value={beautyValues.skinValues.texture} onChange={(v) => updateBeautyValue('skinValues', 'texture', v)} />
                    </AccordionSection>
                    <AccordionSection title="Khuôn mặt AI" isOpen={expandedSections.face} onToggle={() => toggleSection('face')}>
                        <SliderControl label="8. Gọt mặt (Face Slim)" value={beautyValues.faceValues.slim} onChange={(v) => updateBeautyValue('faceValues', 'slim', v)} />
                        <SliderControl label="9. Cằm V-line (V-Line)" value={beautyValues.faceValues.vline} onChange={(v) => updateBeautyValue('faceValues', 'vline', v)} />
                        <SliderControl label="10. Thu nhỏ cằm (Chin Shrink)" value={beautyValues.faceValues.chinShrink} onChange={(v) => updateBeautyValue('faceValues', 'chinShrink', v)} />
                        <SliderControl label="11. Trán (Forehead)" value={beautyValues.faceValues.forehead} min={-100} max={100} onChange={(v) => updateBeautyValue('faceValues', 'forehead', v)} />
                        <SliderControl label="12. Hàm (Jaw)" value={beautyValues.faceValues.jaw} onChange={(v) => updateBeautyValue('faceValues', 'jaw', v)} />
                        <div className="pt-4 border-t border-zinc-800 mt-2">
                            <label className="text-xs text-gray-400 mb-3 block font-medium uppercase tracking-wide">13. Mũi AI (Nose)</label>
                            <SliderControl label="Thu gọn mũi" value={beautyValues.faceValues.noseSlim} onChange={(v) => updateBeautyValue('faceValues', 'noseSlim', v)} />
                            <SliderControl label="Nâng sống mũi" value={beautyValues.faceValues.noseBridge} onChange={(v) => updateBeautyValue('faceValues', 'noseBridge', v)} />
                        </div>
                    </AccordionSection>
                    <AccordionSection title="Mắt" isOpen={expandedSections.eyes} onToggle={() => toggleSection('eyes')}>
                        <SliderControl label="Làm to mắt" value={beautyValues.eyeValues.enlarge} onChange={(v) => updateBeautyValue('eyeValues', 'enlarge', v)} />
                        <SliderControl label="Xóa thâm mắt" value={beautyValues.eyeValues.darkCircle} onChange={(v) => updateBeautyValue('eyeValues', 'darkCircle', v)} />
                        <SliderControl label="Độ sâu mắt" value={beautyValues.eyeValues.depth} onChange={(v) => updateBeautyValue('eyeValues', 'depth', v)} />
                        <div className="mt-4 space-y-4">
                            <div className="pt-2 border-t border-zinc-800">
                                <label className="text-xs text-gray-400 block mb-3 font-medium">Thay màu lens</label>
                                <div className="grid grid-cols-4 gap-2">
                                    <LensSwatch label="Không" color="transparent" isEmpty isActive={beautyValues.eyeMakeup.lens === 'none'} onClick={() => onBeautyChange(prev => ({...prev, eyeMakeup: {...prev.eyeMakeup, lens: 'none'}}))} />
                                    <LensSwatch label="Nâu tự nhiên" color="#8d6e63" isActive={beautyValues.eyeMakeup.lens === 'natural_brown'} onClick={() => onBeautyChange(prev => ({...prev, eyeMakeup: {...prev.eyeMakeup, lens: 'natural_brown'}}))} />
                                    <LensSwatch label="Nâu lạnh" color="#5d4037" isActive={beautyValues.eyeMakeup.lens === 'cool_brown'} onClick={() => onBeautyChange(prev => ({...prev, eyeMakeup: {...prev.eyeMakeup, lens: 'cool_brown'}}))} />
                                    <LensSwatch label="Ghi xám" color="#9e9e9e" isActive={beautyValues.eyeMakeup.lens === 'gray'} onClick={() => onBeautyChange(prev => ({...prev, eyeMakeup: {...prev.eyeMakeup, lens: 'gray'}}))} />
                                    <LensSwatch label="Xanh khói" color="#607d8b" isActive={beautyValues.eyeMakeup.lens === 'smoky_blue'} onClick={() => onBeautyChange(prev => ({...prev, eyeMakeup: {...prev.eyeMakeup, lens: 'smoky_blue'}}))} />
                                 </div>
                            </div>
                        </div>
                    </AccordionSection>
                    <AccordionSection title="Miệng" isOpen={expandedSections.mouth} onToggle={() => toggleSection('mouth')}>
                         <div className="space-y-4">
                             <SliderControl label="Nụ cười AI" value={beautyValues.mouthValues.smile} onChange={(v) => updateBeautyValue('mouthValues', 'smile', v)} />
                         </div>
                         <div className="mt-5 pt-4 border-t border-zinc-800">
                             <div className="flex justify-between items-center mb-3"><label className="text-xs text-gray-400 font-medium">Son môi AI</label><Smile size={14} className="text-fuchsia-500 opacity-70" /></div>
                             <div className="grid grid-cols-3 gap-2">
                                 <ColorOptionButton label="Không" colorCode="#27272a" textColor="text-gray-400" isActive={beautyValues.lipstick === 'none'} onClick={() => onBeautyChange(prev => ({...prev, lipstick: 'none'}))} />
                                 <ColorOptionButton label="Nude hồng" colorCode="#f8bbd0" textColor="text-gray-800" isActive={beautyValues.lipstick === 'nude_pink'} onClick={() => onBeautyChange(prev => ({...prev, lipstick: 'nude_pink'}))} />
                                 <ColorOptionButton label="Hồng đất" colorCode="#d81b60" isActive={beautyValues.lipstick === 'earthy_pink'} onClick={() => onBeautyChange(prev => ({...prev, lipstick: 'earthy_pink'}))} />
                                 <ColorOptionButton label="Đỏ cherry" colorCode="#d50000" isActive={beautyValues.lipstick === 'cherry_red'} onClick={() => onBeautyChange(prev => ({...prev, lipstick: 'cherry_red'}))} />
                                 <ColorOptionButton label="Đỏ rượu" colorCode="#880e4f" isActive={beautyValues.lipstick === 'wine_red'} onClick={() => onBeautyChange(prev => ({...prev, lipstick: 'wine_red'}))} />
                                 <ColorOptionButton label="Cam san hô" colorCode="#ff7043" isActive={beautyValues.lipstick === 'coral'} onClick={() => onBeautyChange(prev => ({...prev, lipstick: 'coral'}))} />
                             </div>
                         </div>
                    </AccordionSection>
                </div>
            )}

            {/* === FILTER TAB === */}
            {activeTab === TabType.FILTER && (
                <div className="space-y-6">
                    <div className="flex gap-2 overflow-x-auto pb-1 custom-scrollbar">
                        {FILTER_CATEGORIES.map(cat => (
                            <button
                                key={cat.id}
                                onClick={() => onFilterChange(prev => ({ ...prev, selectedCategory: cat.id }))}
                                className={`px-4 py-2 rounded-full text-xs font-semibold border transition-all whitespace-nowrap
                                    ${filterValues.selectedCategory === cat.id ? 'bg-fuchsia-600 text-white border-transparent' : 'border-zinc-700 text-gray-400 hover:text-white'}
                                `}
                            >
                                {cat.label}
                            </button>
                        ))}
                    </div>

                    <div className="grid grid-cols-3 gap-3">
                        {visibleFilters.slice(0, 15).map(preset => (
                            <FilterPresetCard 
                                key={preset.id}
                                preset={preset}
                                isActive={filterValues.selectedPreset === preset.id}
                                onSelect={() => onFilterChange(prev => ({ ...prev, selectedPreset: preset.id }))}
                                intensity={filterValues.intensity}
                            />
                        ))}
                    </div>

                    <SliderControl 
                        label="Cường độ bộ lọc" 
                        value={filterValues.intensity} 
                        onChange={(v) => onFilterChange(prev => ({ ...prev, intensity: v }))} 
                    />

                    <button
                        onClick={() => onFilterChange(prev => ({ ...prev, selectedPreset: null }))}
                        className="w-full text-xs text-gray-400 hover:text-white transition-colors py-2"
                    >
                        Gỡ bộ lọc
                    </button>
                </div>
            )}

            {/* === EFFECT TAB === */}
            {activeTab === TabType.EFFECT && (
                <div className="space-y-5">
                    <EffectCard title="Bokeh Overlay" description="PNG bokeh preset + slider cường độ" icon={Sun}>
                        <div className="flex gap-2 flex-wrap">
                            {BOKEH_PRESETS.map(preset => (
                                <button
                                    key={preset.id}
                                    onClick={() => onEffectsChange(prev => ({ ...prev, bokeh: { ...prev.bokeh, preset: preset.id } }))}
                                    className={`px-3 py-2 rounded-xl text-xs border ${effectsValues.bokeh.preset === preset.id ? 'border-fuchsia-500 bg-fuchsia-500/10 text-white' : 'border-zinc-700 text-gray-400 hover:text-white'}`}
                                >
                                    {preset.label}
                                </button>
                            ))}
                        </div>
                        <SliderControl 
                            label="Cường độ bokeh" 
                            value={effectsValues.bokeh.intensity} 
                            onChange={(v) => onEffectsChange(prev => ({ ...prev, bokeh: { ...prev.bokeh, intensity: v } }))} 
                        />
                    </EffectCard>

                    <EffectCard title="Light leak" description="Overlay ảnh leak, Screen/Add mode" icon={Sparkles}>
                        <div className="flex gap-2 flex-wrap">
                            {LIGHT_LEAK_PRESETS.map(preset => (
                                <button
                                    key={preset.id}
                                    onClick={() => onEffectsChange(prev => ({ ...prev, lightLeak: { ...prev.lightLeak, preset: preset.id } }))}
                                    className={`px-3 py-2 rounded-xl text-xs border ${effectsValues.lightLeak.preset === preset.id ? 'border-fuchsia-500 bg-fuchsia-500/10 text-white' : 'border-zinc-700 text-gray-400 hover:text-white'}`}
                                >
                                    {preset.label}
                                </button>
                            ))}
                        </div>
                        <SliderControl 
                            label="Cường độ leak" 
                            value={effectsValues.lightLeak.intensity} 
                            onChange={(v) => onEffectsChange(prev => ({ ...prev, lightLeak: { ...prev.lightLeak, intensity: v } }))} 
                        />
                    </EffectCard>

                    <EffectCard title="Film Grain" description="Noise Gaussian & blend nhẹ" icon={Film}>
                        <SliderControl 
                            label="Độ mạnh hạt film"
                            value={effectsValues.filmGrain}
                            onChange={(v) => onEffectsChange(prev => ({ ...prev, filmGrain: v }))}
                        />
                    </EffectCard>

                    <EffectCard title="Vignette" description="Mặt nạ radial gradient" icon={Scan}>
                        <SliderControl 
                            label="Độ tối viền"
                            value={effectsValues.vignette}
                            onChange={(v) => onEffectsChange(prev => ({ ...prev, vignette: v }))}
                        />
                    </EffectCard>

                    <EffectCard title="Blur (Portrait Blur)" description="Tách người & làm mờ nền" icon={Waves}>
                        <SliderControl 
                            label="Độ mờ nền"
                            value={effectsValues.portraitBlur}
                            onChange={(v) => onEffectsChange(prev => ({ ...prev, portraitBlur: v }))}
                        />
                    </EffectCard>
                </div>
            )}

            {/* === AI PRO TAB === */}
            {activeTab === TabType.AI_PRO && (
                <div className="space-y-4">
                    <div className="rounded-3xl bg-gradient-to-br from-[#47236a] via-[#2c1f43] to-[#120d1c] border border-fuchsia-500/40 p-5 shadow-lg shadow-purple-900/40 text-sm">
                        <div className="flex items-start justify-between gap-4">
                            <div>
                                <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-fuchsia-200/70 mb-1">AI PRO Nâng cao</p>
                                <h3 className="text-base font-semibold text-white">Mario AI Studio</h3>
                                <p className="text-[11px] text-fuchsia-100/70 mt-1">Tự động hóa quy trình chỉnh sửa phức tạp với sức mạnh AI.</p>
                            </div>
                            <div className="w-12 h-12 rounded-2xl bg-white/10 flex items-center justify-center text-fuchsia-100">
                                <Sparkles size={22} />
                            </div>
                        </div>
                    </div>

                    <div className="space-y-3">
                        {AI_SECTIONS.map(section => {
                            const isOpen = aiSectionsOpen[section.id];
                            return (
                                <div key={section.id} className="rounded-3xl border border-zinc-800 bg-[#111013] overflow-hidden">
                                    <button
                                        type="button"
                                        onClick={() => toggleAiSection(section.id)}
                                        className="w-full flex items-center justify-between gap-4 px-4 py-3"
                                    >
                                        <div className="text-left text-xs">
                                            <p className="font-semibold text-white">{section.title}</p>
                                            {section.subtitle && <p className="text-[11px] text-gray-500">{section.subtitle}</p>}
                                        </div>
                                        <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
                                    </button>
                                    {isOpen && (
                                        <div className="px-4 pb-4 space-y-2">
                                            {section.modules.map(module => {
                                                const isActive = selectedAIModule === module.id;
                                                return (
                                                    <button
                                                        type="button"
                                                        key={module.id}
                                                        onClick={() => handleSelectAIModule(module.id)}
                                                        className={`w-full rounded-2xl px-4 py-3 text-left border transition-all text-xs
                                                            ${isActive ? 'border-fuchsia-500/80 bg-fuchsia-500/10 shadow-lg shadow-fuchsia-900/30' : 'border-transparent bg-zinc-900/50 hover:border-zinc-700'}
                                                        `}
                                                    >
                                                        <div className="flex items-start justify-between gap-3">
                                                            <div>
                                                                <p className="font-semibold text-white">{module.title}</p>
                                                                <p className="text-[11px] text-gray-400 mt-0.5">{module.description}</p>
                                                            </div>
                                                            {isActive && (
                                                                <span className="w-6 h-6 rounded-full border border-fuchsia-300 text-fuchsia-200 flex items-center justify-center text-[11px] font-bold">
                                                                    <Check size={12} />
                                                                </span>
                                                            )}
                                                        </div>
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>

                    {selectedModuleData && (
                        <div className="rounded-3xl border border-fuchsia-500/30 bg-[#111014] p-5 space-y-4 text-sm">
                            <div className="flex items-start justify-between gap-4">
                                <div>
                                    <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-fuchsia-200/60">AI Control Panel</p>
                                    <h3 className="text-base font-semibold text-white mt-1">{selectedModuleData.title}</h3>
                                    <p className="text-[11px] text-gray-400">{selectedModuleData.description}</p>
                                </div>
                                <div className="text-right">
                                    <p className="text-[10px] uppercase text-gray-500">Trạng thái</p>
                                    <p className={`text-xs font-semibold ${AI_STATUS_CLASSES[selectedModuleStatus]}`}>
                                        {selectedModuleStatus === 'error' && aiProErrors?.[selectedAIModule || '']
                                            ? aiProErrors[selectedAIModule || '']
                                            : AI_STATUS_LABELS[selectedModuleStatus]}
                                    </p>
                                </div>
                            </div>
                            {!isSceneGenerationModule && (
                                <>
                                    <SliderControl 
                                        label="Cường độ"
                                        value={aiIntensity}
                                        min={0}
                                        max={100}
                                        onChange={(value) => setAiIntensity(value)}
                                    />
                                    {requiresReferenceImage && (
                                        <div className="space-y-3 rounded-2xl border border-fuchsia-500/20 bg-fuchsia-500/5 p-4">
                                            <div className="space-y-2">
                                                <label className="text-[11px] text-gray-300 font-medium">Ảnh tham chiếu (Ảnh A)</label>
                                                <input
                                                    type="file"
                                                    accept="image/*"
                                                    onChange={handleReferenceImageChange}
                                                    className="w-full bg-zinc-900 border border-zinc-700 rounded-xl px-3 py-2 text-sm text-gray-100 file:mr-4 file:py-1 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-fuchsia-600 file:text-white hover:file:bg-fuchsia-700 file:cursor-pointer focus:border-fuchsia-500 focus:ring-1 focus:ring-fuchsia-500 outline-none"
                                                />
                                                {referenceImagePreview && (
                                                    <div className="mt-2 rounded-lg overflow-hidden border border-zinc-700">
                                                        <img
                                                            src={referenceImagePreview}
                                                            alt="Reference preview"
                                                            className="w-full h-32 object-cover"
                                                        />
                                                    </div>
                                                )}
                                            </div>
                                            <p className="text-[10px] text-gray-500 leading-relaxed">
                                                Tải lên ảnh tham chiếu để sao chép bảng màu và tone màu. Ảnh hiện tại sẽ được áp dụng màu từ ảnh tham chiếu.
                                            </p>
                                        </div>
                                    )}
                                </>
                            )}
                            {isSceneGenerationModule && (
                                <div className="space-y-3 rounded-2xl border border-fuchsia-500/20 bg-fuchsia-500/5 p-4">
                                    <div className="space-y-2">
                                        <label className="text-[11px] text-gray-300 font-medium">Trang phục mong muốn</label>
                                        <input
                                            type="text"
                                            value={sceneInputs.outfit}
                                            onChange={(e) => setSceneInputs(prev => ({ ...prev, outfit: e.target.value }))}
                                            placeholder="VD: Vest đen lịch lãm"
                                            className="w-full bg-zinc-900 border border-zinc-700 rounded-xl px-3 py-2 text-sm text-gray-100 placeholder:text-gray-500 focus:border-fuchsia-500 focus:ring-1 focus:ring-fuchsia-500 outline-none"
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-[11px] text-gray-300 font-medium">Địa điểm / Bối cảnh</label>
                                        <input
                                            type="text"
                                            value={sceneInputs.location}
                                            onChange={(e) => setSceneInputs(prev => ({ ...prev, location: e.target.value }))}
                                            placeholder="VD: Tháp Eiffel lúc hoàng hôn"
                                            className="w-full bg-zinc-900 border border-zinc-700 rounded-xl px-3 py-2 text-sm text-gray-100 placeholder:text-gray-500 focus:border-fuchsia-500 focus:ring-1 focus:ring-fuchsia-500 outline-none"
                                        />
                                    </div>
                                    <p className="text-[10px] text-gray-500 leading-relaxed">
                                        Gemini sẽ giữ nguyên khuôn mặt hiện tại, sau đó vẽ lại trang phục và bối cảnh theo mô tả trên. 
                                        Prompt chuẩn được tối ưu sẵn cho chất lượng cao, ánh sáng cinematic.
                                    </p>
                                </div>
                            )}
                            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                                <p className="text-[11px] text-gray-500 max-w-[65%]">Điều chỉnh mức ảnh hưởng và nhấn “Tạo ngay” để chạy mô-đun đã chọn.</p>
                                <button
                                    type="button"
                                    onClick={handleAIGenerate}
                                    disabled={isGenerateDisabled}
                                    className={`flex items-center justify-center gap-2 px-4 py-2 rounded-2xl text-xs font-semibold transition-all
                                        ${isGenerateDisabled 
                                            ? 'bg-zinc-800 text-gray-500 border border-zinc-700 cursor-not-allowed'
                                            : 'bg-gradient-to-r from-fuchsia-600 to-purple-600 text-white border border-transparent hover:shadow-lg hover:shadow-fuchsia-900/30'
                                        }
                                    `}
                                >
                                    {(isAIGenerating || selectedModuleStatus === 'running') ? (
                                        <>
                                            <Loader2 size={16} className="animate-spin" />
                                            <span>Đang tạo...</span>
                                        </>
                                    ) : (
                                        <span>Tạo ngay (Generate)</span>
                                    )}
                                </button>
                                {missingSceneInputs && (
                                    <p className="text-[10px] text-fuchsia-300 mt-1">Điền đầy đủ thông tin trang phục và địa điểm.</p>
                                )}
                                {missingSceneSource && (
                                    <p className="text-[10px] text-fuchsia-300 mt-1">Vui lòng tải một ảnh lên Canvas trước.</p>
                                )}
                                {missingReference && (
                                    <p className="text-[10px] text-fuchsia-300 mt-1">Vui lòng tải ảnh tham chiếu để sử dụng tính năng này.</p>
                                )}
                                {sceneError && (
                                    <p className="text-[10px] text-red-400 mt-1">{sceneError}</p>
                                )}
                            </div>
                            {activeAIInsight && (
                                <div className="rounded-2xl border border-zinc-800/60 bg-zinc-900/40 p-4 space-y-2">
                                    <div className="flex items-center justify-between">
                                        <p className="text-xs font-semibold text-gray-200 uppercase tracking-wide">Kết quả AI</p>
                                        <span className="text-[10px] text-gray-500">{activeAIMetricLabel}</span>
                                    </div>
                                    <p className="text-[11px] text-gray-400">{activeAIInsight.summary}</p>
                                    {!!activeAIInsight.steps?.length && (
                                        <ul className="list-disc list-inside text-[11px] text-gray-500 space-y-1">
                                            {activeAIInsight.steps.slice(0, 3).map((step, index) => (
                                                <li key={index}>{step}</li>
                                            ))}
                                        </ul>
                                    )}
                                    {activeAIInsight.qaNotes && (
                                        <p className="text-[10px] text-gray-500 border-t border-zinc-800 pt-2">{activeAIInsight.qaNotes}</p>
                                    )}
                                </div>
                            )}
                            {selectedPreview?.preview && selectedAIModule && (
                                <div className={`rounded-2xl border ${previewStyle.border} bg-[#0f1317] p-4 space-y-4`}>
                                    <div className="flex items-start justify-between gap-4">
                                        <div>
                                            <p className="text-[10px] font-semibold uppercase tracking-[0.25em] text-gray-400">Xem trước kết quả</p>
                                            <p className="text-[11px] text-gray-400 mt-1">
                                                {previewMeta?.description ?? 'Ảnh PNG nền trong suốt từ mô-đun tách nền.'}
                                            </p>
                                        </div>
                                        {previewMeta?.badge ? (
                                            <span className={`text-[10px] font-semibold px-3 py-1 rounded-full ${previewStyle.chipBg} ${previewStyle.chipText}`}>
                                                {previewMeta.badge}
                                            </span>
                                        ) : (
                                            <span className={`text-[11px] font-semibold ${previewStyle.accentText}`}>Preview</span>
                                        )}
                                    </div>
                                    <div className="text-[10px] text-gray-500">{previewDescription}</div>
                                    <div
                                        className="w-full rounded-2xl overflow-hidden border border-zinc-800"
                                        style={{
                                            backgroundImage:
                                                'linear-gradient(45deg, rgba(255,255,255,0.08) 25%, transparent 25%),' +
                                                'linear-gradient(-45deg, rgba(255,255,255,0.08) 25%, transparent 25%),' +
                                                'linear-gradient(45deg, transparent 75%, rgba(255,255,255,0.08) 75%),' +
                                                'linear-gradient(-45deg, transparent 75%, rgba(255,255,255,0.08) 75%)',
                                            backgroundSize: '24px 24px',
                                            backgroundPosition: '0 0, 0 12px, 12px -12px, -12px 0',
                                        }}
                                    >
                                        <img
                                            src={selectedPreview.preview}
                                            alt="AI PRO cutout preview"
                                            className="w-full h-full object-contain mix-blend-normal"
                                        />
                                    </div>
                                    <div className="flex flex-col gap-3">
                                        <div className="flex flex-wrap items-center gap-3">
                                            {selectedPreview?.preview && (
                                                <a
                                                    href={selectedPreview.preview}
                                                    download={`${selectedAIModule}-preview.png`}
                                                    className="inline-flex items-center gap-1 text-[11px] text-cyan-300 underline decoration-dotted hover:text-cyan-100"
                                                >
                                                    {previewDownloadLabel}
                                                </a>
                                            )}
                                            {selectedPreview?.mask && (
                                                <a
                                                    href={selectedPreview.mask}
                                                    download={`${selectedAIModule}-mask.png`}
                                                    className="inline-flex items-center gap-1 text-[11px] text-emerald-300 underline decoration-dotted hover:text-emerald-200"
                                                >
                                                    Tải mask PNG
                                                </a>
                                            )}
                                        </div>
                                        <div className="flex flex-col w-full gap-2 sm:flex-row sm:justify-end">
                                            <button
                                                type="button"
                                                onClick={() => onDismissAiPreview?.(selectedAIModule)}
                                                className="w-full rounded-2xl border border-zinc-700 px-4 py-2 text-xs font-semibold text-gray-200 hover:border-zinc-500 transition-colors sm:w-auto"
                                            >
                                                Giữ ảnh gốc
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => onApplyAiPreview?.(selectedAIModule)}
                                                className="w-full rounded-2xl px-4 py-2 text-xs font-semibold text-white bg-gradient-to-r from-emerald-600 to-teal-500 hover:shadow-lg hover:shadow-emerald-900/30 transition-all sm:w-auto"
                                            >
                                                Áp dụng kết quả
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
  
          </div>
      </aside>
    );
  };
