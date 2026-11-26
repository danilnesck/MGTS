import io
import csv
import logging
import os
import uuid
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload
from app.utils import get_gps_coordinates

from app.database import get_db
from app.models import Detection, Inspection
from app.schemas import InspectionResponse
from app.ml.predictor import ModelNotReady, predict_image

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 МБ

# =====================================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ОТРИСОВКИ РАМОК
# =====================================================================
def draw_boxes_on_image(image: Image.Image, detections: list) -> Image.Image:
    """Рисует рамки и подписи на изображении."""
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        # Красный для дефектов, зеленый для нормы
        color = "red" if det.needs_repair else "green"
        
        # Рисуем рамку
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Подписываем текст над рамкой
        text = f"{det.object_type}: {det.status}"
        text_bbox = draw.textbbox((x1, y1 - 15), text)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 15), text, fill="white")
        
    return image

# =====================================================================
# 1. АНАЛИЗ ИЗОБРАЖЕНИЯ (POST)
# =====================================================================
@router.post("/analyze", response_model=InspectionResponse, summary="Загрузить фото и проанализировать")
def analyze_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # 1. Читаем и валидируем размер файла
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Файл пустой")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Файл слишком большой (макс 10 МБ)")

    # 2. Проверяем, что это действительно картинка
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Некорректный файл изображения")

    # 3. Отправляем в ML
    try:
        detections = predict_image(image)
    except ModelNotReady:
        raise HTTPException(
            status_code=503,
            detail="Модель анализа пока не подключена. Ожидайте обновления ML-модуля.",
        )
    except Exception as e:
        logger.exception("Ошибка при обработке изображения моделью")
        raise HTTPException(status_code=500, detail="Не удалось обработать изображение")

    # 4. Сохраняем оригинал на диск
    safe_filename = (file.filename or "upload").replace("\\", "/").split("/")[-1][:255]
    unique_id = uuid.uuid4()
    
    os.makedirs("uploads", exist_ok=True) # Создаем папку на всякий случай
    orig_filename = f"{unique_id}_orig.jpg"
    orig_path = f"uploads/{orig_filename}"
    
    with open(orig_path, "wb") as buffer:
        buffer.write(contents)

    # 5. Рисуем рамки (если что-то нашли) и сохраняем результат
    processed_url = None
    if detections:
        processed_image = draw_boxes_on_image(image.copy(), detections)
        proc_filename = f"{unique_id}_proc.jpg"
        proc_path = f"uploads/{proc_filename}"
        processed_image.save(proc_path, format="JPEG", quality=90)
        # Формируем URL, по которому FastAPI будет раздавать эту картинку
        processed_url = f"/uploads/{proc_filename}" 

        # 6. Вытаскиваем GPS-координаты из EXIF (если есть)
    lat, lon = get_gps_coordinates(contents)

    # 7. Сохраняем результаты в БД
    inspection = Inspection(
        image_filename=safe_filename,
        processed_image_url=processed_url,
        detections=[
            Detection(
                object_type=item.object_type,
                status=item.status,
                description=item.description,
                confidence=item.confidence,
                needs_repair=item.needs_repair,
                bbox=item.bbox,
                latitude=lat,  # <-- Передаем координаты
                longitude=lon, # <-- Передаем координаты
            )
            for item in detections
        ],
    )

# =====================================================================
# 2. ПОЛУЧИТЬ СПИСОК ВСЕХ ПРОВЕРОК (GET)
# =====================================================================
@router.get("/inspections", response_model=list[InspectionResponse], summary="Получить историю всех проверок")
def get_all_inspections(db: Session = Depends(get_db)):
    inspections = db.query(Inspection).options(joinedload(Inspection.detections)).order_by(Inspection.created_at.desc()).all()
    return inspections


# =====================================================================
# 3. ПОЛУЧИТЬ ДЕТАЛИ ОДНОЙ ПРОВЕРКИ (GET)
# =====================================================================
@router.get("/inspections/{inspection_id}", response_model=InspectionResponse, summary="Получить детали проверки по ID")
def get_inspection(inspection_id: int, db: Session = Depends(get_db)):
    inspection = db.query(Inspection).options(joinedload(Inspection.detections)).filter(Inspection.id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="Инспекция с таким ID не найдена")
    return inspection


# =====================================================================
# 4. ЭКСПОРТ ОТЧЕТА В CSV (Excel) (GET)
# =====================================================================
@router.get("/report/csv", summary="Выгрузить отчет по дефектам в CSV (Excel)")
def export_report(db: Session = Depends(get_db)):
    """
    Формирует CSV файл (разделитель точка с запятой для русского Excel).
    Включает только те объекты, которым требуется ремонт (needs_repair=True).
    """
    defects = db.query(Detection).options(joinedload(Detection.inspection)).filter(Detection.needs_repair == True).all()
    
    output = io.StringIO()
    writer = csv.writer(output, delimiter=';') 
    
        # В функции export_report в заголовке добавь:
    writer.writerow(["ID фото", "Объект", "Статус", "Описание", "Уверенность", "Широта", "Долгота"])
    
    # И внутри цикла for defect in defects:
    writer.writerow([
        defect.inspection.image_filename,
        defect.object_type,
        defect.status,
        defect.description or "Описание отсутствует",
        f"{defect.confidence:.2f}",
        defect.latitude or "",
        defect.longitude or ""
    ])