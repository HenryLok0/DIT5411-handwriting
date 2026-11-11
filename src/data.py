"""Dataset split, TFRecord writing, and tf.data readers.

The assignment uses the first 40 photos of each character for training and the
rest for testing. Training photos are then expanded to at least 200 samples
per character. Testing photos are never augmented.
"""

from __future__ import annotations

import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.augment import render_sample

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
CONTAINER_NAMES = {"cleaned_data", "data", "images", "img", "train", "test"}
SHARD_SIZE = 100_000
Entry = object  # pathlib.Path or zipfile.ZipInfo


def natural_key(name: str):
    """Sort 'char_2' before 'char_10'."""
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", name)]


def decode_zip_filename(info: zipfile.ZipInfo) -> str:
    """Decode a zip member name. The published archive stores names in Big5."""
    name = info.filename
    if info.flag_bits & 0x800:
        return name.replace("\\", "/")
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name.replace("\\", "/")
    for encoding in ("big5", "cp950", "gbk", "utf-8"):
        try:
            return raw.decode(encoding).replace("\\", "/")
        except UnicodeDecodeError:
            continue
    return name.replace("\\", "/")


def class_and_file(decoded_name: str):
    """Return (character, filename) or None when the member is not an image."""
    parts = [part for part in decoded_name.split("/") if part and part not in (".", "..")]
    if not parts:
        return None
    filename = parts[-1]
    suffix = Path(filename).suffix.lower()
    if suffix not in IMG_EXT:
        return None
    if len(parts) >= 2 and parts[-2].lower() not in CONTAINER_NAMES:
        character = parts[-2]
    else:
        character = Path(filename).stem.split("_")[0]
    if not character or character.lower() in CONTAINER_NAMES:
        return None
    return character, filename


def read_gray_file(path: Path) -> np.ndarray:
    """Read a grayscale image. np.fromfile keeps non-ASCII Windows paths working."""
    import cv2

    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def read_gray_bytes(raw: bytes) -> np.ndarray:
    import cv2

    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Cannot decode image bytes")
    return image


def scan_folder(root: Path) -> Dict[str, List[Tuple[str, Path]]]:
    """Group images by character folder, or by the prefix before '_' if flat."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {root}")

    groups: Dict[str, List[Tuple[str, Path]]] = {}
    subdirs = [path for path in root.iterdir() if path.is_dir()]
    if subdirs:
        for folder in subdirs:
            images = [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMG_EXT]
            if not images:
                continue
            groups[folder.name] = [(path.name, path) for path in images]
        if groups:
            return groups

    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMG_EXT:
            continue
        character = path.stem.split("_")[0]
        groups.setdefault(character, []).append((path.name, path))
    if not groups:
        raise FileNotFoundError(f"No images found in {root}.")
    return groups


def index_zip(zip_file: zipfile.ZipFile) -> Dict[str, List[Tuple[str, zipfile.ZipInfo]]]:
    """Group zip members by character. File bytes are not read here."""
    groups: Dict[str, List[Tuple[str, zipfile.ZipInfo]]] = {}
    for info in zip_file.infolist():
        if info.is_dir():
            continue
        decoded = decode_zip_filename(info)
        parsed = class_and_file(decoded)
        if parsed is None:
            continue
        character, filename = parsed
        groups.setdefault(character, []).append((filename, info))
    if not groups:
        raise FileNotFoundError("The archive has no readable images. If class names look garbled, check Big5 decoding.")
    return groups


def split_samples(entries: Sequence[Tuple[str, Entry]]) -> Tuple[List[Entry], List[Entry]]:
    """First 40 photos train, the remainder test. Fewer than 41 photos use an 80/20 cut."""
    ordered = [entry for _, entry in sorted(entries, key=lambda item: natural_key(item[0]))]
    count = len(ordered)
    if count >= 41:
        return ordered[:40], ordered[40:]
    train_count = max(1, int(round(count * 0.8)))
    if train_count >= count and count > 1:
        train_count = count - 1
    return ordered[:train_count], ordered[train_count:]


def versions_per_image(train_count: int, target: int) -> int:
    """How many geometric versions each real training photo needs."""
    if train_count <= 0:
        raise ValueError("train_count must be positive")
    return max(1, int(math.ceil(target / train_count)))


def summarize_groups(groups: Dict[str, Sequence]) -> dict:
    counts = [len(items) for items in groups.values()]
    total = int(sum(counts))
    return {
        "classes": len(counts),
        "images": total,
        "min_per_class": int(min(counts)) if counts else 0,
        "max_per_class": int(max(counts)) if counts else 0,
        "mean_per_class": float(total / len(counts)) if counts else 0.0,
        "classes_fewer_than_41": int(sum(1 for count in counts if count < 41)),
    }


def _example(image: np.ndarray, label: int) -> bytes:
    import tensorflow as tf

    features = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
    }
    return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()


class _ShardedWriter:
    def __init__(self, directory: Path, prefix: str):
        import tensorflow as tf

        self.directory = directory
        self.prefix = prefix
        self.count = 0
        self.shard = -1
        self.writer = None
        self.files: List[str] = []
        self._tf = tf

    def write(self, payload: bytes) -> None:
        if self.writer is None or (self.count > 0 and self.count % SHARD_SIZE == 0):
            self._rotate()
        self.writer.write(payload)
        self.count += 1

    def _rotate(self) -> None:
        self.close()
        self.shard += 1
        path = self.directory / f"{self.prefix}-{self.shard:05d}.tfrecord"
        self.files.append(path.name)
        options = self._tf.io.TFRecordOptions(compression_type="GZIP")
        self.writer = self._tf.io.TFRecordWriter(str(path), options=options)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def write_tfrecords(
    output_dir: Path,
    img_size: int,
    work_size: int,
    target_per_class: int,
    seed: int,
    max_classes: int = 0,
    zip_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    index_only: bool = False,
) -> dict:
    """Index the dataset, split each character, and store augmented training images."""
    output_dir = Path(output_dir)
    rng = np.random.RandomState(seed)
    zip_file = zipfile.ZipFile(zip_path) if zip_path else None
    failed = 0
    try:
        if zip_file is not None:
            groups = index_zip(zip_file)

            def loader(entry):
                return read_gray_bytes(zip_file.read(entry))

        elif data_dir is not None:
            groups = scan_folder(Path(data_dir))

            def loader(entry):
                return read_gray_file(entry)

        else:
            raise ValueError("--zip or --data-dir is required.")

        names = sorted(groups)
        if max_classes and max_classes > 0:
            print(f"Warning: using only the first {max_classes} characters after sorting. This is not the 13,065-class result required by the assignment.")
            names = names[:max_classes]
            groups = {name: groups[name] for name in names}

        stats = summarize_groups(groups)
        print(
            f"Index complete: {stats['classes']} characters, {stats['images']} images, "
            f"{stats['min_per_class']}–{stats['max_per_class']} per character (mean {stats['mean_per_class']:.2f})."
        )
        print(f"Characters with fewer than 41 images, split about 80/20 instead: {stats['classes_fewer_than_41']}")
        preview = names[:8]
        print("Class examples: " + ", ".join(preview))
        if stats["classes"] != 13065 and not max_classes:
            print("Warning: class count is not 13,065. The full dataset should have 13,065 characters so the output layer has 13,065 neurons.")

        if index_only:
            stats["index_only"] = True
            return stats

        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"{output_dir} is not empty. Use another output folder, or delete the old files first.")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Shuffle class order on disk so a modest training shuffle still mixes characters.
        write_order = list(names)
        rng.shuffle(write_order)
        labels_path = output_dir / "labels.txt"
        labels_path.write_text("\n".join(names) + "\n", encoding="utf-8")
        label_of = {name: index for index, name in enumerate(names)}

        train_writer = _ShardedWriter(output_dir, "train")
        test_writer = _ShardedWriter(output_dir, "test")
        train_real = 0
        try:
            for class_index, name in enumerate(write_order, start=1):
                train_entries, test_entries = split_samples(groups[name])
                if not train_entries:
                    continue
                copies = versions_per_image(len(train_entries), target_per_class)
                label = label_of[name]
                for entry in train_entries:
                    try:
                        source = loader(entry)
                    except Exception:
                        failed += 1
                        continue
                    train_real += 1
                    for aug_id in range(copies):
                        rendered = render_sample(source, aug_id, img_size, work_size)
                        train_writer.write(_example(rendered, label))
                for entry in test_entries:
                    try:
                        source = loader(entry)
                    except Exception:
                        failed += 1
                        continue
                    rendered = render_sample(source, 0, img_size, work_size)
                    test_writer.write(_example(rendered, label))
                if class_index % 200 == 0 or class_index == len(write_order):
                    print(
                        f"Processed {class_index}/{len(write_order)} characters, "
                        f"training samples {train_writer.count}, test samples {test_writer.count}, read failures {failed}"
                    )
        finally:
            train_writer.close()
            test_writer.close()

        if train_writer.count == 0 or test_writer.count == 0:
            raise RuntimeError("The training or test TFRecord is empty. Check the data with --index-only first.")

        meta = {
            "img_size": img_size,
            "work_size": work_size,
            "num_classes": len(names),
            "samples_per_class_target": target_per_class,
            "train_examples": train_writer.count,
            "test_examples": test_writer.count,
            "train_real_images": train_real,
            "failed_images": failed,
            "train_files": train_writer.files,
            "test_files": test_writer.files,
            "seed": seed,
            "compression": "GZIP",
            "augmentation": "opencv rotation, shear, scale; aug_id 0 is the original photo",
            **stats,
        }
        (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Write complete: {output_dir}")
        return meta
    finally:
        if zip_file is not None:
            zip_file.close()


def load_meta(records_dir: Path) -> dict:
    path = Path(records_dir) / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {path}. Run prepare_records.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(records_dir: Path) -> List[str]:
    path = Path(records_dir) / "labels.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {path}.")
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def make_record_dataset(records_dir: Path, split: str, batch_size: int, shuffle: bool, seed: int, max_classes: int = 0):
    """Read prepared TFRecords. Images are already augmented and resized."""
    import tensorflow as tf

    meta = load_meta(records_dir)
    img_size = int(meta["img_size"])
    filenames = [str(Path(records_dir) / name) for name in meta[f"{split}_files"]]
    missing = [name for name in filenames if not os.path.isfile(name)]
    if missing:
        raise FileNotFoundError("Missing TFRecord: " + ", ".join(missing[:3]))

    dataset = tf.data.TFRecordDataset(
        filenames,
        compression_type="GZIP",
        num_parallel_reads=tf.data.AUTOTUNE,
        buffer_size=8 * 1024 * 1024,
    )
    # Shuffle serialized records. Decoding first would pin a large float buffer in RAM.
    if shuffle:
        dataset = dataset.shuffle(100_000, seed=seed, reshuffle_each_iteration=True)

    def _parse(example):
        parsed = tf.io.parse_single_example(
            example,
            {
                "image": tf.io.FixedLenFeature([], tf.string),
                "label": tf.io.FixedLenFeature([], tf.int64),
            },
        )
        image = tf.io.decode_raw(parsed["image"], tf.uint8)
        image = tf.reshape(image, (img_size, img_size, 1))
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(parsed["label"], tf.int32)
        return image, label

    dataset = dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if max_classes and max_classes > 0:
        dataset = dataset.filter(lambda _image, label: label < max_classes)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def make_folder_dataset(
    data_dir: Path,
    img_size: int,
    work_size: int,
    target_per_class: int,
    batch_size: int,
    seed: int,
    max_classes: int = 0,
):
    """On-the-fly path for a small folder dataset. The full 13,065-class run should use TFRecords."""
    import tensorflow as tf

    groups = scan_folder(Path(data_dir))
    names = sorted(groups)
    if max_classes and max_classes > 0:
        print(f"Warning: using only the first {max_classes} characters after sorting. This is not the 13,065-class result required by the assignment.")
        names = names[:max_classes]
    label_of = {name: index for index, name in enumerate(names)}

    train_paths: List[str] = []
    train_aug: List[int] = []
    train_labels: List[int] = []
    test_paths: List[str] = []
    test_labels: List[int] = []

    for name in names:
        train_entries, test_entries = split_samples(groups[name])
        if not train_entries:
            continue
        copies = versions_per_image(len(train_entries), target_per_class)
        for entry in train_entries:
            for aug_id in range(copies):
                train_paths.append(str(entry))
                train_aug.append(aug_id)
                train_labels.append(label_of[name])
        for entry in test_entries:
            test_paths.append(str(entry))
            test_labels.append(label_of[name])

    if not train_paths or not test_paths:
        raise RuntimeError("The training or test list is empty.")

    def _loader(paths, aug_ids):
        images = np.empty((len(paths), img_size, img_size, 1), dtype=np.float32)
        for index, (path, aug_id) in enumerate(zip(paths, aug_ids)):
            text = path.decode("utf-8") if isinstance(path, (bytes, bytearray)) else str(path)
            try:
                source = read_gray_file(Path(text))
                rendered = render_sample(source, int(aug_id), img_size, work_size)
            except Exception:
                rendered = np.zeros((img_size, img_size), dtype=np.uint8)
            images[index, :, :, 0] = rendered.astype(np.float32) / 255.0
        return images

    def _train_map(paths, aug_ids, labels):
        images = tf.numpy_function(_loader, [paths, aug_ids], tf.float32)
        images.set_shape((None, img_size, img_size, 1))
        return images, labels

    def _test_map(paths, labels):
        aug_ids = tf.zeros_like(labels)
        images = tf.numpy_function(_loader, [paths, aug_ids], tf.float32)
        images.set_shape((None, img_size, img_size, 1))
        return images, labels

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_paths, np.asarray(train_aug, dtype=np.int32), np.asarray(train_labels, dtype=np.int32))
    )
    train_ds = train_ds.shuffle(len(train_paths), seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size).map(_train_map, num_parallel_calls=1).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, np.asarray(test_labels, dtype=np.int32)))
    test_ds = test_ds.batch(batch_size).map(_test_map, num_parallel_calls=1).prefetch(tf.data.AUTOTUNE)

    meta = {
        "img_size": img_size,
        "num_classes": len(names),
        "labels": names,
        "train_examples": len(train_paths),
        "test_examples": len(test_paths),
        **summarize_groups({name: groups[name] for name in names}),
    }
    return train_ds, test_ds, meta


def iter_first_image(zip_path: Optional[Path] = None, data_dir: Optional[Path] = None):
    """Load one real image for the augmentation preview."""
    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as zip_file:
            groups = index_zip(zip_file)
            name = sorted(groups)[0]
            filename, info = sorted(groups[name], key=lambda item: natural_key(item[0]))[0]
            return name, filename, read_gray_bytes(zip_file.read(info))
    groups = scan_folder(Path(data_dir))
    name = sorted(groups)[0]
    filename, path = sorted(groups[name], key=lambda item: natural_key(item[0]))[0]
    return name, filename, read_gray_file(path)
