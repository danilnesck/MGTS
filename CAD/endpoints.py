import io
import csv
import logging
from pathlib import Path
import uuid
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from sqlalchemy.orm import Session, joinedload

from database import get_db
from model_loader import REVIEW_STATUS
from models import Detection, Inspection
from predictor import ModelNotReady, predict_image
from schemas import InspectionResponse
from utils import get_gps_coordinates

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 МБ
# Keep Pillow's opener when Ultralytics later installs its optional HEIF fallback.
open_upload_image = Image.open

# =====================================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ОТРИСОВКИ РАМОК
# =====================================================================
def draw_boxes_on_image(image: Image.Image, detections: list) -> Image.Image:
    """Рисует рамки и подписи на изображении."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, size=16)
            break
        except OSError:
            continue
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = "red" if det.needs_repair else "green"
        if det.status == REVIEW_STATUS:
            color = "#b06b00"
        
        # Рисуем рамку
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Подписываем текст над рамкой
        text = f"{det.object_type}: {det.status}"
        label_y = max(0, y1 - 20)
        text_bbox = draw.textbbox((x1, label_y), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, label_y), text, fill="white", font=font)
        
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
        with open_upload_image(io.BytesIO(contents)) as source:
            source.verify()
        with open_upload_image(io.BytesIO(contents)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Некорректный файл изображения")

    # 3. Отправляем в ML
    try:
        detections = predict_image(image)
    except ModelNotReady as exc:
        logger.warning("ML недоступен: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Веса анализа изображения отсутствуют или не удалось их загрузить.",
        )
    except Exception:
        logger.exception("Ошибка при обработке изображения моделью")
        raise HTTPException(status_code=500, detail="Не удалось обработать изображение")

    # 4. Сохраняем оригинал на диск
    safe_filename = (file.filename or "upload").replace("\\", "/").split("/")[-1][:255]
    unique_id = uuid.uuid4()
    
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    orig_path = upload_dir / f"{unique_id}_orig.jpg"
    proc_path = upload_dir / f"{unique_id}_proc.jpg"

    lat, lon = get_gps_coordinates(contents)
    inspection = Inspection(
        image_filename=safe_filename,
        processed_image_url=f"/uploads/{proc_path.name}" if detections else None,
        detections=[
            Detection(
                object_type=item.object_type,
                status=item.status,
                description=item.description,
                confidence=item.confidence,
                needs_repair=item.needs_repair,
                bbox=item.bbox,
                latitude=lat,
                longitude=lon,
            )
            for item in detections
        ],
    )
    try:
        orig_path.write_bytes(contents)
        if detections:
            draw_boxes_on_image(image.copy(), detections).save(proc_path, format="JPEG", quality=90)
        db.add(inspection)
        db.commit()
    except Exception:
        db.rollback()
        for path in (orig_path, proc_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Не удалось удалить файл незавершённой проверки: %s", path)
        raise
    db.refresh(inspection)
    return inspection

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
    
    writer.writerow(["ID фото", "Объект", "Статус", "Описание", "Уверенность", "Широта", "Долгота"])
    
    for defect in defects:
        writer.writerow([
            defect.inspection.image_filename,
            defect.object_type,
            defect.status,
            defect.description or "",
            f"{defect.confidence:.2f}",
            defect.latitude if defect.latitude is not None else "",
            defect.longitude if defect.longitude is not None else "",
        ])

    return Response(
        output.getvalue().encode("utf-8-sig"),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="defects.csv"'},
    )
