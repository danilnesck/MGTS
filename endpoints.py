from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from PIL import Image
import io
from app.database import get_db
from app.models import Inspection
from app.schemas import InspectionResponse, DetectedObject
from app.ml.predictor import predict_image

router = APIRouter()

@router.post("/analyze", response_model=list[InspectionResponse])
async def analyze_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Проверка, что это картинка
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # 2. Читаем картинку
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 3. Отправляем в ML (заглушку)
    try:
        detections = predict_image(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML processing error: {str(e)}")
    
    # 4. Сохраняем результаты в БД
    saved_inspections = []
    for det in detections:
        db_item = Inspection(
            image_filename=file.filename,
            object_type=det.object_type,
            status=det.status,
            confidence=det.confidence,
            needs_repair=det.needs_repair,
            bbox=det.bbox,
            # Заглушка для координат (в реальности брать из EXIF или БД)
            latitude=55.7558, 
            longitude=37.6173
        )
        db.add(db_item)
        saved_inspections.append(db_item)
    
    db.commit()
    for item in saved_inspections:
        db.refresh(item)
        
    # 5. Формируем ответ (группируем по картинке)
    # В упрощенном виде возвращаем список всех найденных объектов с привязкой к фото
    response = []
    if saved_inspections:
        response.append(InspectionResponse(
            id=saved_inspections[0].id,
            image_filename=file.filename,
            objects_detected=detections,
            created_at=saved_inspections[0].created_at
        ))
    
    return response