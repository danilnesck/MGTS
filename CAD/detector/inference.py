"""Run detection on the full image and optionally on overlapping views."""
from math import isfinite

import torch
from torchvision.ops import nms


def image_views(image, strategy):
    width, height = image.size
    yield image, (0, 0, width, height)
    if strategy == "full":
        return
    if strategy == "center":
        windows = ((0.15, 0.15, 0.85, 0.85),)
    elif strategy == "tiles":
        windows = (
            (0, 0, 0.62, 0.62), (0.38, 0, 1, 0.62),
            (0, 0.38, 0.62, 1), (0.38, 0.38, 1, 1),
            (0.19, 0.19, 0.81, 0.81),
        )
    else:
        raise ValueError(f"Неизвестный режим детектора: {strategy}")
    for x1, y1, x2, y2 in windows:
        box = round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)
        if box[2] - box[0] >= 32 and box[3] - box[1] >= 32:
            yield image.crop(box), box


def detect(detector, image, *, device, size, threshold, max_detections, strategy="full"):
    views, locations = zip(*image_views(image, strategy), strict=True)
    predicted = detector.predict(
        list(views), device=str(device), imgsz=size, conf=threshold,
        max_det=max_detections, verbose=False,
    )
    candidates = []
    for result, (left, top, right, bottom) in zip(predicted, locations, strict=True):
        patch_width, patch_height = right - left, bottom - top
        boxes = result.boxes.cpu().numpy()
        for (x1, y1, x2, y2), score, cls in zip(boxes.xyxy, boxes.conf, boxes.cls, strict=True):
            if not all(isfinite(value) for value in (x1, y1, x2, y2, score, cls)):
                continue
            if (
                (left > 0 and x1 < 0.02 * patch_width)
                or (top > 0 and y1 < 0.02 * patch_height)
                or (right < image.width and x2 > 0.98 * patch_width)
                or (bottom < image.height and y2 > 0.98 * patch_height)
            ):
                continue
            candidates.append({
                "box_xyxy": [float(x1 + left), float(y1 + top), float(x2 + left), float(y2 + top)],
                "score": float(score),
                "detector_class": result.names[int(cls)],
            })
    if len(views) == 1 or not candidates:
        return candidates
    boxes = torch.tensor([item["box_xyxy"] for item in candidates])
    scores = torch.tensor([item["score"] for item in candidates])
    keep = nms(boxes, scores, 0.50).tolist()[:max_detections]
    return [candidates[index] for index in keep]
