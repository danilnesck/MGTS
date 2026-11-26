# app/schemas.py
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, computed_field
from model_loader import REVIEW_STATUS

class DetectedObject(BaseModel):
    object_type: str
    status: str
    description: str | None = None
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] = Field(min_length=4, max_length=4)
    needs_repair: bool
    latitude: float | None = None
    longitude: float | None = None

    @computed_field
    @property
    def requires_inspection(self) -> bool:
        return self.needs_repair or self.status == REVIEW_STATUS

class DetectionResponse(DetectedObject):
    model_config = ConfigDict(from_attributes=True)

    id: int

class InspectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_filename: str
    processed_image_url: str | None = None
    created_at: datetime
    detections: list[DetectionResponse]
