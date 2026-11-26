from PIL import Image
from app.schemas import DetectedObject

class ModelNotReady(Exception):
    """
    Исключение, которое выбрасывается, когда ML-модели еще не подключены.
    FastAPI поймает его и вернет 503 Service Unavailable.
    """
    pass

def predict_image(image: Image.Image) -> list[DetectedObject]:
    # TODO: Когда придут модели, здесь будет вызов YOLO и CLIP
    # Пока моделей нет, мы честно сообщаем об этом, чтобы не врать пользователю
    
    raise ModelNotReady()