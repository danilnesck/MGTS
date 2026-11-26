from argparse import ArgumentParser
from copy import deepcopy
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

from classifier.crops import ShiftedCrops, training_manifest
from classifier.network import create_classifier, image_transform
from model_loader import CONDITIONS

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    loss = 0.0
    for images, targets in loader:
        logits = model(images.to(device))
        loss += nn.functional.cross_entropy(logits, targets.to(device), reduction="sum").item()
        scores, predictions = logits.softmax(dim=1).max(dim=1)
        rows.extend(zip(targets.tolist(), predictions.tolist(), scores.tolist(), strict=True))
    return rows, loss / len(rows)


def metrics(rows, classes, threshold):
    matrix = [[0] * len(classes) for _ in classes]
    for actual, predicted, _ in rows:
        matrix[actual][predicted] += 1
    per_class = {}
    for idx, label in enumerate(classes):
        true_positive = matrix[idx][idx]
        actual = sum(matrix[idx])
        predicted = sum(row[idx] for row in matrix)
        per_class[label] = {
            "support": actual,
            "precision": true_positive / predicted if predicted else 0.0,
            "recall": true_positive / actual if actual else 0.0,
            "f1": 2 * true_positive / (actual + predicted) if actual + predicted else 0.0,
        }
    accepted = [(a, p) for a, p, score in rows if score >= threshold]
    return {
        "samples": len(rows),
        "classes": classes,
        "confusion_matrix": matrix,
        "by_class": per_class,
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(classes),
        "accuracy": sum(actual == pred for actual, pred, _ in rows) / len(rows),
        "threshold": threshold,
        "accepted": len(accepted),
        "correct_accepted": sum(actual == pred for actual, pred in accepted),
    }


def choose_threshold(rows, normal_idx, minimum=0.5):
    # False declarations of normal are costlier than sending an object to manual review.
    def utility(threshold):
        score = 0
        for actual, predicted, confidence in rows:
            if confidence >= threshold:
                if actual == predicted:
                    score += 1
                else:
                    score -= 3 if predicted == normal_idx else 1
        return score, threshold

    candidates = {minimum, *[score for _, _, score in rows if score >= minimum]}
    candidates = [value for value in candidates if any(score >= value for _, _, score in rows)]
    return max(candidates, key=utility) if candidates else minimum


def main():
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / ".work/dataset/classification")
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "weights/classifier.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--box-jitter", action="store_true")
    parser.add_argument("--val-only", action="store_true")
    args = parser.parse_args()
    if not args.initial.is_file() or args.epochs < 1 or args.batch < 2:
        parser.error("Нужны стартовые веса, epochs >= 1 и batch >= 2")
    torch.manual_seed(42)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform = image_transform()
    augmentation = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
    ]
    if args.box_jitter:
        augmentation.append(transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1))
    train_transform = transforms.Compose([*augmentation, transform])
    train, val, test = (
        datasets.ImageFolder(args.data / split, transform=preprocess)
        for split, preprocess in (("train", train_transform), ("val", transform), ("test", transform))
    )
    classes = train.classes
    if set(classes) != set(CONDITIONS) or any(
        item.class_to_idx != train.class_to_idx for item in (val, test)
    ):
        raise ValueError("Метки классов должны совпадать во всех выборках")
    if args.box_jitter:
        dataset_root = args.data.parent
        manifest = training_manifest(dataset_root)
        train = ConcatDataset([
            train,
            *(ShiftedCrops(dataset_root, train.samples, manifest, train_transform, seed)
              for seed in (117, 233)),
        ])
    train_loader, val_loader, test_loader = (
        DataLoader(item, batch_size=args.batch, shuffle=split == "train")
        for split, item in (("train", train), ("val", val), ("test", test))
    )

    from torchvision.models import mobilenet_v3_small

    backbone = mobilenet_v3_small()
    backbone.load_state_dict(torch.load(args.initial, map_location="cpu", weights_only=True))
    model = create_classifier(len(classes))
    model.features.load_state_dict(backbone.features.state_dict())
    model.classifier[0].load_state_dict(backbone.classifier[0].state_dict())
    model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 0.00015},
        {"params": model.classifier.parameters(), "lr": 0.001},
    ])

    best_score = (-1.0, float("-inf"))
    for epoch in range(args.epochs):
        model.train()
        # Preserve pretrained batch statistics on the small training split.
        for layer in model.features.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()

        rows, loss = evaluate(model, val_loader, device)
        result = metrics(rows, classes, 0.5)
        score = (result["macro_f1"], -loss)
        print(f"Epoch {epoch + 1}: val_macro_f1={score[0]:.3f}, val_loss={loss:.3f}")
        if score > best_score:
            best_score = score
            best_weights = deepcopy(model.state_dict())
            best_epoch = epoch + 1

    model.load_state_dict(best_weights)
    rows, _ = evaluate(model, val_loader, device)
    threshold = choose_threshold(rows, classes.index("normal"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "mobilenet_v3_small", "classes": classes, "threshold": threshold,
        "state_dict": {name: tensor.cpu() for name, tensor in best_weights.items()},
    }, args.output)
    report = {
        "best_epoch": best_epoch,
        "box_jitter": args.box_jitter,
        "val": metrics(rows, classes, threshold),
    }
    if not args.val_only:
        test_rows, _ = evaluate(model, test_loader, device)
        report["test"] = metrics(test_rows, classes, threshold)
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(args.output, report.get("test", report["val"]))


if __name__ == "__main__":
    main()
