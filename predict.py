"""Predict the character in one image with the saved model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.augment import render_sample
from src.data import read_gray_file


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Recognize one handwritten character.")
    parser.add_argument("--model", type=Path, default=Path("models/best_model.keras"))
    parser.add_argument("--labels", type=Path, default=Path("models/labels.txt"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    import tensorflow as tf

    labels = [line for line in args.labels.read_text(encoding="utf-8").splitlines() if line]
    model = tf.keras.models.load_model(args.model)
    img_size = int(model.input_shape[1])
    image = render_sample(read_gray_file(args.image), 0, img_size, img_size)
    batch = image.astype(np.float32).reshape(1, img_size, img_size, 1) / 255.0
    probabilities = model.predict(batch, verbose=0)[0]
    order = np.argsort(probabilities)[::-1][: args.topk]
    print(f"圖片：{args.image}")
    for rank, index in enumerate(order, start=1):
        name = labels[index] if index < len(labels) else f"class_{index}"
        print(f"{rank}. {name}    {probabilities[index]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
