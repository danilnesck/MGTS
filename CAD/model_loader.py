"""Load local model artifacts once per backend worker."""
from functools import lru_cache
from pathlib import Path
from pickle import UnpicklingError
from threading import Lock

from config import settings

OBJECT_TYPES = {"tksh": "ТКШ", "rsh": "РЩ"}
CONDITIONS = {
    "normal": "Удовлетворительно",
    "coating_damage": "Повреждение ЛКП",
    "mechanical_damage": "Механические повреждения",
}
REVIEW_STATUS = "Требуется проверка"


class ModelNotReady(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_models():
    paths = (Path(settings.DETECTOR_WEIGHTS), Path(settings.CLASSIFIER_WEIGHTS))
    for path in paths:
        if not path.is_file():
            raise ModelNotReady(f"Файл весов отсутствует: {path}")

    try:
        import torch
        from ultralytics import YOLO
        from classifier.network import create_classifier, image_transform

        device = settings.ML_DEVICE
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        detector = YOLO(str(paths[0]), task="detect").to(device)
        if set(detector.names.values()) != set(OBJECT_TYPES) or len(detector.names) != len(OBJECT_TYPES):
            raise ValueError("Веса детектора должны соответствовать классам tksh и rsh")

        checkpoint = torch.load(paths[1], map_location="cpu", weights_only=True)
        classes = checkpoint["classes"]
        if len(classes) != len(CONDITIONS) or set(classes) != set(CONDITIONS):
            raise ValueError("Некорректный набор классов классификатора")
        threshold = settings.CLASSIFIER_THRESHOLD
        if threshold is None:
            threshold = float(checkpoint["threshold"])
        crop_context = float(checkpoint.get("crop_context", 0))
        if not 0 <= threshold <= 1:
            raise ValueError("Порог классификатора должен лежать в диапазоне [0, 1]")
        if not 0 <= crop_context <= 0.5:
            raise ValueError("Запас рамки классификатора должен лежать в диапазоне [0, 0.5]")
        architecture = checkpoint["architecture"]
        if architecture == "mobilenet_v3_small":
            classifier = create_classifier(len(classes))
        elif architecture == "feature_linear":
            from classifier.features import FeatureClassifier

            classifier = FeatureClassifier(checkpoint["backbone"], checkpoint["grids"], len(classes))
        else:
            raise ValueError("Неизвестная архитектура классификатора")
        classifier.load_state_dict(checkpoint["state_dict"])
        classifier.to(device).eval()
        return detector, classifier, classes, image_transform(), device, threshold, crop_context
    except (AttributeError, ImportError, KeyError, OSError, RuntimeError, TypeError, UnpicklingError, ValueError) as exc:
        raise ModelNotReady(f"Ошибка загрузки весов ML: {exc}") from exc


load_lock = Lock()


def load_models():
    with load_lock:
        return _load_models()
