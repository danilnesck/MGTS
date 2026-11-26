import io
import logging
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Detection, Inspection
from app.schemas import InspectionResponse
from app.ml.predictor import ModelNotReady, predict_image

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 МБ

# ВАЖНО: здесь def, а не async def!
@router.post("/analyze", response_model=InspectionResponse)
def analyze_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)

    if not contents:
        raise HTTPException(status_code=400, detail="Файл пустой")

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Файл слишком большой (макс 10 МБ)")

    try:
        # Проверяем, что файл действительно является изображением.
        image = Image.open(io.BytesIO(contents))
        image.verify()
        # verify() закрывает объект — открываем заново для дальнейшей работы.
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Некорректный файл изображения")

    try:
        detections = predict_image(image)
    except ModelNotReady:
        raise HTTPException(
            status_code=503,
            detail="Модель анализа пока не подключена",
        )
    except Exception:
        logger.exception("Ошибка при обработке изображения моделью")
        raise HTTPException(status_code=500, detail="Не удалось обработать изображение")

    # Безопасная обработка имени файла
    safe_filename = (file.filename or "upload").replace("\\", "/").split("/")[-1]
    safe_filename = safe_filename[:255] or "upload"

    # Создаем ОДНУ инспекцию и привязываем к ней все найденные объекты
    inspection = Inspection(
        image_filename=safe_filename,
        detections=[
            Detection(
                object_type=item.object_type,
                status=item.status,
                confidence=item.confidence,
                needs_repair=item.needs_repair,
                bbox=item.bbox,
            )
            for item in detections
        ],
    )

    try:
        db.add(inspection)
        db.commit()
        db.refresh(inspection)
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Не удалось сохранить результат анализа")
        raise HTTPException(status_code=500, detail="Не удалось сохранить результат")

    return InspectionResponse.model_validate(inspection)