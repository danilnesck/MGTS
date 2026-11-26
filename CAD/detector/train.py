from argparse import ArgumentParser
import json
import os
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / ".work/dataset/detection.yaml")
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "weights/detector.pt")
    parser.add_argument("--runs", type=Path, default=ROOT / ".work/runs")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--recipe", choices=("original", "small-objects"), default="original")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    if not args.initial.is_file() or not args.data.is_file():
        parser.error("Не найден файл стартовых весов или detection.yaml")

    import torch
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    detector = YOLO(str(args.initial.resolve()), task="detect")
    augmentation = {"degrees": 0, "mosaic": 0.25, "close_mosaic": 8}
    if args.recipe == "small-objects":
        augmentation.update(
            degrees=4, mosaic=0.10, scale=0.30, translate=0.08, cos_lr=True,
            optimizer="AdamW", lr0=0.0012, lrf=0.10, warmup_epochs=2.0,
        )
    detector.train(
        data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, workers=0, patience=args.patience, device=device, amp=False,
        seed=42, plots=False, freeze=args.freeze,
        project=str(args.runs.resolve()), name="detector", cache="ram", **augmentation,
    )
    best = Path(detector.trainer.best)
    result = YOLO(str(best), task="detect").val(
        data=str(args.data.resolve()), split=args.eval_split, device=device, imgsz=args.imgsz,
        batch=args.batch, workers=0, plots=False, project=str(args.runs.resolve()), name=args.eval_split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    copyfile(best, args.output)
    metrics = {
        "split": args.eval_split,
        "image_size": args.imgsz,
        **{key: float(value) for key, value in result.results_dict.items()},
        "by_class": {
            result.names[class_id]: dict(zip(
                ("precision", "recall", "mAP50", "mAP50-95"),
                map(float, result.box.class_result(index)), strict=True
            ))
            for index, class_id in enumerate(result.box.ap_class_index.tolist())
        },
    }
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), "utf-8")
    print(args.output, metrics)


if __name__ == "__main__":
    main()
