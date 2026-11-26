"""ML inference and adaptation to the existing backend response."""
from math import ceil, floor, isfinite
from threading import Lock

from PIL import Image, ImageOps

from config import settings
from model_loader import CONDITIONS, OBJECT_TYPES, REVIEW_STATUS, ModelNotReady, load_models
from schemas import DetectedObject

inference_lock = Lock()


def analyze(image: Image.Image, image_id: str | None = None) -> dict:
    """Return ML1 detections with ML2 results in the supplied image coordinates."""
    image = image.convert("RGB")
    width, height = image.size
    detector, classifier, classes, transform, device, threshold, context = load_models()
    import torch
    from detector.inference import detect

    detections = []
    with inference_lock, torch.inference_mode():
        candidates = detect(
            detector, image, device=device, size=settings.DETECTOR_IMAGE_SIZE,
            threshold=settings.CONFIDENCE_THRESHOLD, max_detections=settings.DETECTOR_MAX_DETECTIONS,
            strategy=settings.DETECTOR_VIEW,
        )
        crops = []
        for candidate in candidates:
            x1, y1, x2, y2 = candidate["box_xyxy"]
            left, top = max(0, floor(x1)), max(0, floor(y1))
            right, bottom = min(width, ceil(x2)), min(height, ceil(y2))
            if left >= right or top >= bottom:
                continue
            box = [left, top, right, bottom]
            views = [transform(image.crop(box))]
            if context:
                horizontal, vertical = (right - left) * context, (bottom - top) * context
                expanded = (
                    max(0, floor(left - horizontal)), max(0, floor(top - vertical)),
                    min(width, ceil(right + horizontal)), min(height, ceil(bottom + vertical)),
                )
                views.append(transform(image.crop(expanded)))
            crops.append(torch.stack(views))
            detections.append({**candidate, "box_xyxy": box})

        for start in range(0, len(crops), 32):
            objects = crops[start:start + 32]
            batch = torch.cat(objects).to(device)
            probabilities = classifier(batch).softmax(dim=1).reshape(len(objects), -1, len(classes))
            scores, indices = probabilities.mean(dim=1).max(dim=1)
            for detection, score, index in zip(
                detections[start:start + len(objects)], scores.tolist(), indices.tolist(), strict=True
            ):
                category = classes[index]
                if not isfinite(score) or score < threshold:
                    status = "needs_review"
                else:
                    status = "normal" if category == "normal" else "damaged"
                detection["classification"] = {
                    "class": category,
                    "score": score if isfinite(score) else None,
                    "status": status,
                    "requires_inspection": status != "normal",
                }
    return {
        "image_id": image_id, "image_width": width, "image_height": height,
        "detections": detections,
    }


def predict_image(image: Image.Image) -> list[DetectedObject]:
    results = []
    for detection in analyze(image)["detections"]:
        condition = detection["classification"]
        review = condition["status"] == "needs_review"
        results.append(DetectedObject(
            object_type=OBJECT_TYPES[detection["detector_class"]],
            status=REVIEW_STATUS if review else CONDITIONS[condition["class"]],
            description="Рекомендуется ручная оценка состояния" if review else None,
            confidence=detection["score"],
            bbox=detection["box_xyxy"],
            needs_repair=condition["status"] == "damaged",
        ))
    return results


if __name__ == "__main__":
    from argparse import ArgumentParser
    import json
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    try:
        with Image.open(args.image) as source:
            result = analyze(ImageOps.exif_transpose(source), args.image.stem)
    except (OSError, ModelNotReady) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
