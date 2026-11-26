from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    JSON, ForeignKey
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base

class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)
    image_filename = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Связь: у одной инспекции много найденных объектов
    detections = relationship(
        "Detection",
        back_populates="inspection",
        cascade="all, delete-orphan",
    )

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(
        Integer,
        ForeignKey("inspections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    object_type = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    needs_repair = Column(Boolean, nullable=False)
    bbox = Column(JSON, nullable=False)  # [x1, y1, x2, y2]

    # Координаты оставляем пустыми (None), пока нет реальных данных
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    inspection = relationship("Inspection", back_populates="detections")