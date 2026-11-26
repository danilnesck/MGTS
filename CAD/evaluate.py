"""Evaluate detector and the end-to-end pipeline on the fixed dataset splits."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from config import ROOT, settings
from dataset import DETECTOR_IDS


def load_cases(dataset, split):
    annotations = json.loads((ROOT / "detector/annotations.json").read_text("utf-8"))
    rows = json.loads((dataset / "manifest.json").read_text("utf-8"))
    for row in rows:
        if row["split"] != split:
            continue
        with Image.open(dataset / "detection/images" / split / row["image"]) as source:
            image = source.convert("RGB")
        boxes = annotations[row["source"]]["boxes"]
        scale = np.array([image.width, image.height] * 2)
        yield image, [{**box, "xyxy": np.asarray(box["xyxy"]) * scale} for box in boxes]


def match(expected, predictions, overlap):
    """Return a reference index or None for each prediction in descending score order."""
    unmatched = set(range(len(expected)))
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["box_xyxy"]
        best = overlap, None
        for index in sorted(unmatched):
            target = expected[index]
            if target["type"] != prediction["detector_class"]:
                continue
            u1, v1, u2, v2 = target["xyxy"]
            intersection = max(0, min(x2, u2) - max(x1, u1)) * max(0, min(y2, v2) - max(y1, v1))
            union = (x2 - x1) * (y2 - y1) + (u2 - u1) * (v2 - v1) - intersection
            iou = intersection / union if union > 0 else 0
            if iou >= overlap and (best[1] is None or iou > best[0]):
                best = iou, index
        index = best[1]
        if index is not None:
            unmatched.remove(index)
        yield index


def detector_metrics(outputs, threshold=None):
    from ultralytics.utils.metrics import ap_per_class

    scores, labels, targets, correct = [], [], [], []
    ious = np.linspace(0.5, 0.95, 10)
    for expected, predictions in outputs:
        predictions = sorted(predictions, key=lambda item: -item["score"])
        scores.extend(item["score"] for item in predictions)
        labels.extend(DETECTOR_IDS[item["detector_class"]] for item in predictions)
        targets.extend(DETECTOR_IDS[item["type"]] for item in expected)
        correct.extend(zip(*(
            (idx is not None for idx in match(expected, predictions, iou)) for iou in ious
        ), strict=True))
    scores = np.asarray(scores)
    correct = np.asarray(correct, dtype=bool).reshape(-1, len(ious))
    ap, classes = ap_per_class(correct, scores, np.asarray(labels), np.asarray(targets))[5:7]

    def operating_point(value):
        selected = scores >= value
        tp = int(correct[selected, 0].sum())
        fp = int(selected.sum()) - tp
        fn = len(targets) - tp
        return {
            "threshold": float(value), "tp": tp, "fp": fp, "fn": fn,
            "precision_at_iou50": tp / max(1, tp + fp),
            "recall_at_iou50": tp / max(1, tp + fn),
            "f1_at_iou50": 2 * tp / max(1, 2 * tp + fp + fn),
        }

    thresholds = (threshold,) if threshold is not None else (step / 40 for step in range(2, 37))
    return {
        "mAP50": float(ap[:, 0].mean()),
        "mAP50-95": float(ap.mean()),
        "by_class": {
            list(DETECTOR_IDS)[idx]: float(ap[i, 0]) for i, idx in enumerate(classes)
        },
        **max(map(operating_point, thresholds), key=lambda item: (item["f1_at_iou50"], item["threshold"])),
    }


def cascade_metrics(outputs):
    counts = Counter(correct=0, wrong=0, review=0, missed=0, false_normal=0)
    for expected, predictions in outputs:
        predictions = sorted(predictions, key=lambda item: -item["score"])
        results = {index: prediction["classification"] for index, prediction in zip(
            match(expected, predictions, 0.5), predictions, strict=True
        ) if index is not None}
        for index, target in enumerate(expected):
            if target["condition"] is None:
                continue
            classification = results.get(index)
            if classification is None:
                counts["missed"] += 1
            elif classification["status"] == "needs_review":
                counts["review"] += 1
            elif classification["class"] == target["condition"]:
                counts["correct"] += 1
            else:
                counts["wrong"] += 1
                if classification["status"] == "normal" and target["condition"] != "normal":
                    counts["false_normal"] += 1
    return dict(counts)


def main():
    from time import perf_counter

    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from ultralytics import YOLO
    from classifier.train import evaluate as evaluate_classifier
    from classifier.train import metrics as classifier_metrics
    from detector.inference import detect
    from model_loader import load_models
    from predictor import analyze

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / ".work/refined")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--detector", type=Path, default=settings.DETECTOR_WEIGHTS)
    parser.add_argument("--classifier", type=Path, default=settings.CLASSIFIER_WEIGHTS)
    parser.add_argument("--size", type=int, default=settings.DETECTOR_IMAGE_SIZE)
    parser.add_argument("--view", choices=("full", "center", "tiles"), default=settings.DETECTOR_VIEW)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--class-threshold", type=float)
    parser.add_argument("--detector-only", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.split == "test" and args.threshold is None:
        parser.error("Для тестовой оценки укажите порог, выбранный на валидации.")
    if args.threshold is not None and not 0.001 <= args.threshold <= 1:
        parser.error("Порог детектора должен быть от 0.001 до 1.")
    if args.class_threshold is not None and not 0 <= args.class_threshold <= 1:
        parser.error("Порог классификатора должен быть от 0 до 1.")
    torch.set_num_threads(4)
    device = settings.ML_DEVICE
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cases = list(load_cases(args.dataset, args.split))
    if not cases:
        parser.error(f"Выборка {args.split} пуста.")
    model = YOLO(str(args.detector), task="detect")
    with torch.inference_mode():
        detected = [(expected, detect(
            model, image, device=device, size=args.size, threshold=0.001,
            max_detections=settings.DETECTOR_MAX_DETECTIONS, strategy=args.view,
        )) for image, expected in cases]
    detector = detector_metrics(detected, args.threshold)
    report = {
        "split": args.split, "images": len(cases), "image_size": args.size, "view": args.view,
        "detector_weights": args.detector.name,
        "detector": detector,
    }
    if not args.detector_only:
        settings.DETECTOR_WEIGHTS, settings.CLASSIFIER_WEIGHTS = args.detector, args.classifier
        settings.DETECTOR_IMAGE_SIZE, settings.DETECTOR_VIEW = args.size, args.view
        settings.CONFIDENCE_THRESHOLD = detector["threshold"]
        if args.class_threshold is not None:
            settings.CLASSIFIER_THRESHOLD = args.class_threshold
        analyze(cases[0][0])
        started = perf_counter()
        states = [(expected, analyze(image)["detections"]) for image, expected in cases]
        report["cascade"] = cascade_metrics(states)
        report["mean_inference_ms"] = round(1000 * (perf_counter() - started) / len(cases), 1)
        report["device"] = device
        _, classifier, classes, preprocess, class_device, class_threshold, context = load_models()
        crops = ImageFolder(args.dataset / "classification" / args.split, transform=preprocess)
        if crops.classes != list(classes):
            raise ValueError("Порядок классов контрольных вырезок не совпадает с весами.")
        rows, _ = evaluate_classifier(classifier, DataLoader(crops, batch_size=12), class_device)
        report["classifier_weights"] = args.classifier.name
        report["classifier_crop_context"] = context
        report["classifier_on_reference_crops"] = classifier_metrics(rows, classes, class_threshold)
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.out:
        args.out.write_text(serialized + "\n", "utf-8")


if __name__ == "__main__":
    main()
