from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PointModel(BaseModel):
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)


class SkinValues(BaseModel):
    smooth: float = 0
    whiten: float = 0
    even: float = 0
    korean: float = 0
    texture: float = 50


class AcneMode(BaseModel):
    auto: bool = False
    manualPoints: List[PointModel] = Field(default_factory=list)


class FaceValues(BaseModel):
    slim: float = 0
    vline: float = 0
    chinShrink: float = 0
    forehead: float = 0
    jaw: float = 0
    noseSlim: float = 0
    noseBridge: float = 0


class EyeValues(BaseModel):
    enlarge: float = 0
    darkCircle: float = 0
    depth: float = 0
    eyelid: float = 0


class EyeMakeup(BaseModel):
    eyeliner: bool = False
    lens: Literal[
        "none",
        "natural_brown",
        "cool_brown",
        "gray",
        "smoky_blue",
    ] = "none"


class MouthValues(BaseModel):
    smile: float = 0
    volume: float = 0
    heart: float = 0
    teethWhiten: float = 0


class HairValues(BaseModel):
    smooth: float = 0
    volume: float = 0
    shine: float = 0


class BeautyConfig(BaseModel):
    skinMode: Literal["natural", "strong"] = "natural"
    faceMode: Literal["natural"] = "natural"
    skinValues: SkinValues = SkinValues()
    acneMode: AcneMode = AcneMode()
    faceValues: FaceValues = FaceValues()
    eyeValues: EyeValues = EyeValues()
    eyeMakeup: EyeMakeup = EyeMakeup()
    mouthValues: MouthValues = MouthValues()
    lipstick: Literal[
        "none",
        "nude_pink",
        "earthy_pink",
        "cherry_red",
        "wine_red",
        "coral",
    ] = "none"
    hairValues: HairValues = HairValues()
    hairColor: Literal["original"] = "original"


class FaceLandmark(BaseModel):
    x: float
    y: float


class FaceMeta(BaseModel):
    bbox: Optional[List[int]] = None  # [x, y, w, h]
    confidence: Optional[float] = None
    landmarks: Optional[List[FaceLandmark]] = None


class BeautyResponse(BaseModel):
    image: str
    faceMeta: Optional[FaceMeta] = None


class FaceAnalysisResponse(BaseModel):
    faceMeta: Optional[FaceMeta] = None


class SkinBrightenRequest(BaseModel):
    whiten: float = Field(ge=0, le=100, description="Độ sáng da từ 0-100")
    preserveTexture: bool = Field(default=True, description="Giữ nguyên kết cấu da")
    adaptiveMode: bool = Field(default=True, description="Chế độ sáng hóa thích ứng")


class SkinBrightenResponse(BaseModel):
    image: str
    faceMeta: Optional[FaceMeta] = None


class AIProResponse(BaseModel):
    """Response model for AI Pro module endpoints"""
    previewImage: Optional[str] = None
    maskImage: Optional[str] = None
    previewMeta: Optional[Dict[str, Any]] = None
    adjustments: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None

