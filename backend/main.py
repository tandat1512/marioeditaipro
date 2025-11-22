from __future__ import annotations

# Setup Python path TRƯỚC KHI import các module khác
# Điều này cho phép relative imports hoạt động khi chạy từ thư mục backend
import sys
from pathlib import Path

# Setup Python path để relative imports hoạt động khi chạy từ thư mục backend
backend_dir = Path(__file__).parent.resolve()
parent_dir = backend_dir.parent.resolve()
current_working_dir = Path.cwd().resolve()

# Nếu đang ở trong thư mục backend, thêm thư mục cha vào path
# Điều này cho phép Python nhận backend như một package và relative imports hoạt động
if current_working_dir == backend_dir:
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

import json
import logging
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from beauty_pipeline import BeautyPipeline
from config import get_settings
from models import AIProResponse, BeautyConfig, BeautyResponse, FaceAnalysisResponse, SkinBrightenResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Beauty Editor Backend", version="1.0.0")
settings = get_settings()
pipeline = BeautyPipeline()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


async def _load_image(upload: UploadFile) -> np.ndarray:
    """Load image from UploadFile and convert to numpy array."""
    try:
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty image payload")
        array = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Unsupported image format")
        return image
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error loading image: {exc}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(exc)}") from exc


@app.post("/api/beauty/analyze", response_model=FaceAnalysisResponse)
async def analyze_face(image: UploadFile = File(...)) -> FaceAnalysisResponse:
    """Phân tích khuôn mặt trong ảnh."""
    try:
        img = await _load_image(image)
        meta = pipeline.analyze(img)
        if not meta:
            raise HTTPException(status_code=422, detail="Không phát hiện khuôn mặt hợp lệ")
        return FaceAnalysisResponse(faceMeta=meta)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error analyzing face: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi phân tích khuôn mặt: {str(exc)}") from exc


@app.post("/api/beauty/apply", response_model=BeautyResponse)
async def apply_beauty(
    image: UploadFile = File(...),
    beautyConfig: str = Form(...),
) -> BeautyResponse:
    """Áp dụng các hiệu ứng làm đẹp lên ảnh."""
    try:
        config = BeautyConfig.model_validate_json(beautyConfig)
    except json.JSONDecodeError as exc:
        logger.error(f"Invalid beautyConfig JSON: {exc}")
        raise HTTPException(status_code=400, detail=f"Invalid beautyConfig JSON: {exc}") from exc
    except ValidationError as exc:
        logger.error(f"Invalid beautyConfig validation: {exc}")
        raise HTTPException(status_code=400, detail=f"Invalid beautyConfig: {exc.errors()}") from exc
    except Exception as exc:
        logger.error(f"Error parsing beautyConfig: {exc}")
        raise HTTPException(status_code=400, detail=f"Error parsing beautyConfig: {str(exc)}") from exc

    try:
        img = await _load_image(image)
        processed, meta = pipeline.apply(img, config)
        data_url = pipeline.encode_image(processed)
        return BeautyResponse(image=data_url, faceMeta=meta)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error applying beauty effects: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi áp dụng hiệu ứng làm đẹp: {str(exc)}") from exc


@app.post("/api/beauty/brighten-skin", response_model=SkinBrightenResponse)
async def brighten_skin(
    image: UploadFile = File(...),
    whiten: float = Form(50, ge=0, le=100),
    preserveTexture: bool = Form(True),
    adaptiveMode: bool = Form(True),
) -> SkinBrightenResponse:
    """
    Endpoint chuyên dụng cho tính năng Sáng da nâng cao.
    Sử dụng thuật toán Frequency Separation và Adaptive Brightening để sáng hóa da tự nhiên.
    
    Args:
        image: Ảnh đầu vào
        whiten: Độ sáng da (0-100)
        preserveTexture: Giữ nguyên kết cấu da
        adaptiveMode: Chế độ sáng hóa thích ứng (sáng hóa vùng tối nhiều hơn)
    
    Returns:
        SkinBrightenResponse với ảnh đã xử lý
    """
    try:
        img = await _load_image(image)
        
        # Tạo config chỉ với whiten
        from .models import SkinValues, BeautyConfig
        skin_values = SkinValues(whiten=whiten)
        config = BeautyConfig(skinValues=skin_values)
        
        # Áp dụng xử lý
        processed, meta = pipeline.apply(img, config)
        data_url = pipeline.encode_image(processed)
        
        return SkinBrightenResponse(image=data_url, faceMeta=meta)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error brightening skin: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi làm sáng da: {str(exc)}") from exc


# TODO: AI Pro module endpoint - requires implementation of:
# - ai_pro_engine
# - background_remover
# - quality_enhancer
# - color_clone_engine
# - CUTOUT_MODULE_TARGET, QUALITY_MODULE_TARGET, COLOR_MATCH_MODULES constants
@app.post("/api/ai-pro/run", response_model=AIProResponse)
async def run_ai_pro_module(
    moduleId: str = Form(...),
    intensity: int = Form(75),
    options: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    reference: Optional[UploadFile] = File(None),
) -> AIProResponse:
    """
    AI Pro module endpoint - Currently disabled until implementation is complete.
    """
    raise HTTPException(
        status_code=501,
        detail="AI Pro module is not yet implemented. Please use /api/beauty/apply endpoint instead."
    )



