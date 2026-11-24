from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

# Ответ от ML-модели
class DetectedObject(BaseModel):
    object_type: str
    status: str
    confidence: float
    bbox: List[float]
    needs_repair: bool

# Ответ API
class InspectionResponse(BaseModel):
    id: int
    image_filename: str
    objects_detected: List[DetectedObject]
    created_at: datetime

    class Config:
        from_attributes = True