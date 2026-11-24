from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base

class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)
    image_filename = Column(String, index=True)
    object_type = Column(String) # "cable_manhole", "double_purpose_support"
    status = Column(String)      # "normal", "unsatisfactory"
    confidence = Column(Float)
    needs_repair = Column(Boolean, default=False)
    # В реальном проекте лучше использовать PostGIS (Geometry), 
    # но для хакатона сойдут обычные Float (если есть GPS в EXIF)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    bbox = Column(JSON)          # [x1, y1, x2, y2]
    created_at = Column(DateTime(timezone=True), server_default=func.now())