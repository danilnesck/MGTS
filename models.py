# app/models.py
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    JSON, ForeignKey, Text
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)
    image_filename = Column(String(255), nullable=False)
    processed_image_url = Column(String(255), nullable=True) # <-- НОВОЕ ПОЛЕ
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    detections = relationship("Detection", back_populates="inspection", cascade="all, delete-orphan")

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id", ondelete="CASCADE"), nullable=False, index=True)

    object_type = Column(String(100), nullable=False) 
    status = Column(String(100), nullable=False) 
    description = Column(Text, nullable=True) 
    confidence = Column(Float, nullable=False)
    needs_repair = Column(Boolean, nullable=False)
    bbox = Column(JSON, nullable=False) 

    inspection = relationship("Inspection", back_populates="detections")