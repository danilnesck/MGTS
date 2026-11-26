"""Train linear heads with grouped cross-validation on the training split."""
from argparse import ArgumentParser
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from classifier.crops import ShiftedCrops, training_manifest
from classifier.features import BACKBONES, FeatureClassifier
from classifier.network import image_transform
from classifier.train import choose_threshold, metrics

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def extract(model, dataset, device):
    features, targets = [], []
    for images, labels in DataLoader(dataset, batch_size=12):
        features.append(model.embed(images.to(device)).cpu())
        targets.append(labels)
    return torch.cat(features), torch.cat(targets)


def linear_head(features, labels, classes, regularization):
    head = nn.Linear(features.shape[1], classes)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=40, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(head(features), labels)
        loss = loss + regularization * head.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return head


@torch.no_grad()
def score(head, features, labels):
    logits = head(features)
    confidence, prediction = logits.softmax(dim=1).max(dim=1)
    rows = list(zip(labels.tolist(), prediction.tolist(), confidence.tolist(), strict=True))
    return rows, nn.functional.cross_entropy(logits, labels).item()


def cross_validate(features, labels, groups, classes, regularization):
    rows = []
    n = len(groups)
    for group in sorted(set(groups)):
        fold = torch.tensor([name == group for name in groups])
        train = (~fold).repeat(len(features) // n)
        head = linear_head(features[train], labels[train], len(classes), regularization)
        fold_rows, _ = score(head, features[:n][fold], labels[:n][fold])
        rows.extend(fold_rows)
    return metrics(rows, classes, 0)["macro_f1"]


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / ".work/refined")
    parser.add_argument("--pretrained", type=Path, default=ROOT / ".work/pretrained")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--box-jitter", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base = args.dataset / "classification"
    preprocess = image_transform()
    train = datasets.ImageFolder(base / "train", transform=preprocess)
    val = datasets.ImageFolder(base / "val", transform=preprocess)
    classes = train.classes
    if val.class_to_idx != train.class_to_idx:
        raise ValueError("Метки классов обучения и валидации не совпадают")
    train_rows = training_manifest(args.dataset)
    groups = [train_rows[Path(path).stem.split("-")[0]]["group"] for path, _ in train.samples]

    jitter = datasets.ImageFolder(base / "train", transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(3, translate=(0.06, 0.06), scale=(0.92, 1.05)),
        preprocess,
    ]))
    views = [train, jitter]
    if args.box_jitter:
        views.extend(
            ShiftedCrops(args.dataset, train.samples, train_rows, preprocess, seed)
            for seed in (117, 233)
        )
    initial = torch.load(
        args.pretrained / BACKBONES[args.backbone][2], map_location="cpu", weights_only=True
    )
    best = None
    trials = []
    for grids in ((1,), (1, 2)):
        model = FeatureClassifier(args.backbone, grids, len(classes), initial).to(device).eval()
        features, labels = (torch.cat(tensors) for tensors in zip(*(
            extract(model, view, device) for view in views
        ), strict=True))
        val_features, val_labels = extract(model, val, device)
        for penalty in (0.002, 0.01, 0.04):
            cv_f1 = cross_validate(features, labels, groups, classes, penalty)
            head = linear_head(features, labels, len(classes), penalty)
            rows, loss = score(head, val_features, val_labels)
            val_f1 = metrics(rows, classes, 0)["macro_f1"]
            trial = {"grids": list(grids), "l2": penalty, "train_cv_macro_f1": cv_f1, "val_macro_f1": val_f1}
            trials.append(trial)
            print(trial, flush=True)
            rank = (cv_f1, -loss)
            if best is None or rank > best[0]:
                best = rank, grids, model, head, rows
    _, grids, model, head, rows = best
    threshold = choose_threshold(rows, classes.index("normal"), minimum=1 / len(classes))
    model.head.load_state_dict(head.state_dict())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "feature_linear",
        "backbone": args.backbone,
        "grids": list(grids),
        "classes": classes,
        "threshold": threshold,
        "state_dict": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
    }, args.output)
    report = {
        "backbone": args.backbone, "box_jitter": args.box_jitter,
        "train_cv_macro_f1": best[0][0],
        "val": metrics(rows, classes, threshold), "trials": trials,
    }
    if args.test:
        test = datasets.ImageFolder(base / "test", transform=preprocess)
        test_rows, _ = score(head, *extract(model, test, device))
        report["test"] = metrics(test_rows, classes, threshold)
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), "utf-8")


if __name__ == "__main__":
    main()
