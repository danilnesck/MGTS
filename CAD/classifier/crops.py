from pathlib import Path
from random import Random
import json

from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


def training_manifest(dataset):
    manifest = json.loads((dataset / "manifest.json").read_text("utf-8"))
    return {Path(item["image"]).stem: item for item in manifest if item["split"] == "train"}


class ShiftedCrops(Dataset):
    """Train on realistic crop errors around annotated objects."""
    def __init__(self, root, samples, manifest, transform, seed):
        self.root = root / "detection/images/train"
        self.transform = transform
        annotations = json.loads((ROOT / "detector/annotations.json").read_text("utf-8"))
        rng = Random(seed)
        self.samples = []
        for path, label in samples:
            stem, index = Path(path).stem.split("-")
            row = manifest[stem]
            x1, y1, x2, y2 = annotations[row["source"]]["boxes"][int(index)]["xyxy"]
            width, height = x2 - x1, y2 - y1
            box = (
                max(0, x1 + rng.uniform(-0.18, 0.12) * width),
                max(0, y1 + rng.uniform(-0.18, 0.12) * height),
                min(1, x2 + rng.uniform(-0.12, 0.18) * width),
                min(1, y2 + rng.uniform(-0.12, 0.18) * height),
            )
            self.samples.append((row["image"], box, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, (x1, y1, x2, y2), label = self.samples[index]
        with Image.open(self.root / name) as image:
            crop = image.crop((
                x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height,
            )).convert("RGB")
        return self.transform(crop), label
