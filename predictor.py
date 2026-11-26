# app/ml/predictor.py
import random
from PIL import Image
from app.schemas import DetectedObject

class ModelNotReady(Exception):
    pass

def predict_image(image: Image.Image) -> list[DetectedObject]:
    # ВНИМАНИЕ: Это демо-режим для проверки визуализации.
    # Когда придут реальные модели, удали этот блок и напиши вызов нейросетей.
    
    results = []
    # Имитируем, что нашли 2 объекта на фото
    mock_data = [
        {"type": "ТКШ", "status": "Повреждение ЛКП", "desc": "Коррозия корпуса", "repair": True},
        {"type": "РЩ", "status": "Удовлетворительно", "desc": "Дефектов нет", "repair": False}
    ]
    
    width, height = image.size
    
    for item in mock_data:
        # Рисуем рандомные рамки в пределах картинки
        x1 = random.randint(10, width // 2)
        y1 = random.randint(10, height // 2)
        x2 = x1 + random.randint(100, 300)
        y2 = y1 + random.randint(100, 300)
        
        results.append(DetectedObject(
            object_type=item["type"],
            status=item["status"],
            description=item["desc"],
            confidence=0.88,
            bbox=[float(x1), float(y1), float(x2), float(y2)],
            needs_repair=item["repair"]
        ))
    return results