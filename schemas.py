from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

class DetectedObject(BaseModel):
    object_type: str
    status: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] = Field(min_length=4, max_length=4)
    needs_repair: bool

class DetectionResponse(DetectedObject):
    id: int

class InspectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_filename: str
    created_at: datetime
    detections: list[DetectionResponse]