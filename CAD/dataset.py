from argparse import ArgumentParser
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
ARCHIVES = ("doc-1795639519.zip", "doc-1795639569.zip")
FOLDERS = {
    "Удовлетворительное состояние ТКШ": ("tksh", "normal"),
    "Неудовлетворительное состояние ЛКП. ТКШ": ("tksh", "coating_damage"),
    "МП. ТКШ": ("tksh", "mechanical_damage"),
    "Удовлетворительное состояние РЩ": ("rsh", "normal"),
    "Неудовлетворительное состояние ЛКП. РЩ": ("rsh", "coating_damage"),
    "МП.РЩ": ("rsh", "mechanical_damage"),
}
DETECTOR_IDS = {"tksh": 0, "rsh": 1}


@dataclass(frozen=True)
class Photo:
    archive: ZipFile
    member: str
    path: PurePosixPath

    @property
    def label(self):
        return FOLDERS.get(self.path.parent.name)

    def content(self):
        return self.archive.read(self.member)

    def image(self, data=None):
        with Image.open(io.BytesIO(self.content() if data is None else data)) as img:
            return ImageOps.exif_transpose(img).convert("RGB")


def inventory(stack):
    photos = []
    books = []
    for filename in ARCHIVES:
        archive = stack.enter_context(ZipFile(ROOT / filename))
        for member in archive.namelist():
            path = PurePosixPath(member.replace(chr(92), "/"))
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                photos.append(Photo(archive, member, path))
            elif path.suffix.lower() == ".xlsx":
                books.append((archive, member, path))
    order = list(FOLDERS)
    photos.sort(key=lambda p: (
        order.index(p.path.parent.name) if p.label else len(order), p.path.name.casefold()
    ))
    if len({photo.path.as_posix() for photo in photos}) != len(photos):
        raise ValueError("Имена изображений в архивах пересекаются")
    return photos, books


def inspect(photos, books, out):
    from openpyxl import load_workbook

    out.mkdir(parents=True, exist_ok=True)
    report = []
    counts = Counter()
    for index, photo in enumerate(photos, 1):
        data = photo.content()
        report.append({
            "number": index, "path": photo.path.as_posix(), "label": photo.label,
            "sha256": sha256(data).hexdigest(), "size": photo.image(data).size,
        })
        counts[photo.label] += 1
    (out / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(dict(counts))
    photo_keys = Counter((photo.path.parent, photo.path.stem.casefold()) for photo in photos if photo.label)
    row_keys = Counter()
    for archive, member, path in books:
        workbook = load_workbook(io.BytesIO(archive.read(member)), read_only=True, data_only=True)
        try:
            print(path)
            for row in workbook.active.values:
                if any(value is not None for value in row):
                    print(row)
                    name = row[0]
                    if name is not None and str(name).strip().casefold() != "имя файла":
                        if isinstance(name, float) and name.is_integer():
                            name = int(name)
                        row_keys[path.parent, PurePosixPath(str(name).strip()).stem.casefold()] += 1
        finally:
            workbook.close()
    print("Фото с уникальной строкой XLSX:", sum(
        count == 1 and row_keys[key] == 1 for key, count in photo_keys.items()
    ))
    print("Фото без строки XLSX:", list((photo_keys - row_keys).elements()))
    print("Строки XLSX без фото:", list((row_keys - photo_keys).elements()))

    for start in range(0, len(photos), 12):
        sheet = Image.new("RGB", (4 * 260, 3 * 220), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, photo in enumerate(photos[start:start + 12]):
            img = photo.image()
            img.thumbnail((248, 196))
            x, y = offset % 4 * 260 + 6, offset // 4 * 220 + 18
            sheet.paste(img, (x, y))
            draw.text((x, y - 15), f"{start + offset + 1} | {img.width} x {img.height}", fill="black")
        sheet.save(out / f"sheet-{start + 1:02}.jpg", quality=92)


def choose_splits(items):
    from random import Random

    groups = {}
    total = Counter()
    for photo, annotation in items:
        groups.setdefault(annotation["group"], Counter())[photo.label] += 1
        total[photo.label] += 1

    rng = Random(42)
    keys = sorted(groups)
    holdout = max(1, round(len(keys) * 0.16))
    best = None
    for _ in range(4000):
        rng.shuffle(keys)
        parts = {
            "val": keys[:holdout],
            "test": keys[holdout:2 * holdout],
            "train": keys[2 * holdout:],
        }
        cost = 0
        for split, group_ids in parts.items():
            counts = sum((groups[group] for group in group_ids), Counter())
            ratio = 0.68 if split == "train" else 0.16
            cost += sum(
                (counts[label] - count * ratio) ** 2 / count + 100 * (counts[label] == 0)
                for label, count in total.items()
            )
        if best is None or cost < best[0]:
            best = cost, {group: split for split, ids in parts.items() for group in ids}
    if best[0] >= 100:
        raise ValueError("Невозможно распределить группы по выборкам с покрытием всех категорий")
    return best[1]


def focus_small_objects(image, boxes):
    """Yield contextual close-ups of small objects for detector training."""
    from math import ceil, floor

    width, height = image.size
    for target in boxes:
        x1, y1, x2, y2 = target["xyxy"]
        if max(x2 - x1, y2 - y1) > 0.32:
            continue
        # Keep surrounding pole and nearby fittings while enlarging the object.
        fraction = 0.46 if target["type"] == "rsh" else 0.54
        left = min(max((x1 + x2 - fraction) / 2, 0), 1 - fraction)
        top = min(max((y1 + y2 - fraction) / 2, 0), 1 - fraction)
        pixel_box = (
            floor(left * width), floor(top * height),
            ceil((left + fraction) * width), ceil((top + fraction) * height),
        )
        left, top, right, bottom = (
            pixel_box[0] / width, pixel_box[1] / height,
            pixel_box[2] / width, pixel_box[3] / height,
        )
        focused = []
        for box in boxes:
            x1, y1, x2, y2 = box["xyxy"]
            u1, v1, u2, v2 = max(x1, left), max(y1, top), min(x2, right), min(y2, bottom)
            if u1 >= u2 or v1 >= v2:
                continue
            if (u2-u1)*(v2-v1) < 0.3*(x2-x1)*(y2-y1):
                continue
            focused.append({
                **box,
                "xyxy": (
                    (u1-left)/(right-left), (v1-top)/(bottom-top),
                    (u2-left)/(right-left), (v2-top)/(bottom-top),
                ),
            })
        yield image.crop(pixel_box), focused


def save_detection_sample(image, boxes, image_dir, label_dir, stem):
    lines = []
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        lines.append(
            f"{DETECTOR_IDS[box['type']]} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} "
            f"{x2-x1:.6f} {y2-y1:.6f}"
        )
    image.thumbnail((1280, 1280))
    image.save(image_dir / f"{stem}.jpg", quality=95)
    (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", "utf-8")


def prepare(photos, out, focus_small=False):
    from math import ceil, floor

    annotations = json.loads((ROOT / "detector" / "annotations.json").read_text("utf-8"))
    labeled = {photo.path.as_posix(): photo for photo in photos if photo.label}
    if set(annotations) != set(labeled):
        raise ValueError("Разметка не соответствует составу категорий в архивах")
    items = [(photo, annotations[path]) for path, photo in labeled.items()]
    partitions = choose_splits(items)
    hashes = {}
    manifest = []
    for index, (photo, annotation) in enumerate(items, 1):
        group = annotation["group"]
        split = partitions[group]
        data = photo.content()
        digest = sha256(data).hexdigest()
        if hashes.setdefault(digest, split) != split:
            raise ValueError(f"Дубликат изображения в разных выборках: {photo.path}")
        image = photo.image(data)
        width, height = image.size
        image_name = f"{index:03}.jpg"
        for box_index, box in enumerate(annotation["boxes"]):
            kind, condition = box["type"], box["condition"]
            x1, y1, x2, y2 = box["xyxy"]
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"Некорректная рамка: {photo.path} {box}")
            if condition is not None:
                if (kind, condition) != photo.label:
                    raise ValueError(f"Состояние не совпадает с категорией фото: {photo.path}")
                crop = image.crop((
                    floor(x1 * width), floor(y1 * height),
                    ceil(x2 * width), ceil(y2 * height),
                ))
                crop.thumbnail((768, 768))
                crop_dir = out / "classification" / split / condition
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop.save(crop_dir / f"{index:03}-{box_index}.jpg", quality=95)
        det_dir = out / "detection"
        image_dir = det_dir / "images" / split
        label_dir = det_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        if focus_small and split == "train":
            for crop_idx, (crop, boxes) in enumerate(focus_small_objects(image, annotation["boxes"])):
                save_detection_sample(crop, boxes, image_dir, label_dir, f"{index:03}-focus-{crop_idx}")
        save_detection_sample(image, annotation["boxes"], image_dir, label_dir, f"{index:03}")
        manifest.append({
            "source": photo.path.as_posix(), "image": image_name, "sha256": digest,
            "type": photo.label[0], "condition": photo.label[1], "group": group, "split": split,
        })

    # JSON is valid YAML and avoids quoting platform-specific paths by hand.
    (out / "detection.yaml").write_text(json.dumps({
        "path": (out / "detection").resolve().as_posix(),
        **{split: f"images/{split}" for split in ("train", "val", "test")},
        "names": list(DETECTOR_IDS),
    }, indent=2), "utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print(dict(Counter(row["split"] for row in manifest)))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=("inspect", "prepare"), default="inspect")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--focus-small", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        photos, books = inventory(stack)
        if args.mode == "inspect":
            inspect(photos, books, args.out or ROOT / ".work" / "inspection")
        else:
            out = args.out or ROOT / ".work" / "dataset"
            if out.exists() and any(out.iterdir()):
                parser.error(f"Выходная папка должна быть пустой: {out}")
            prepare(photos, out, args.focus_small)
