"""Train the three networks and compare test-set recognition rate.

Recognition rate is top-1 accuracy on the untouched test photos. The test set
is scored once, after the last epoch, so it is not used to pick an epoch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def configure_gpu() -> None:
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def completed_epochs(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = sum(1 for _ in handle) - 1
    return max(0, rows)


def save_history_plot(csv_path: Path, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = []
    accuracy = []
    loss = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            epochs.append(int(float(row["epoch"])) + 1)
            accuracy.append(float(row["accuracy"]))
            loss.append(float(row["loss"]))
    if not epochs:
        return
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, accuracy, marker="o", label="train accuracy")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Train accuracy")
    axis.set_ylim(0, 1)
    twin = axis.twinx()
    twin.plot(epochs, loss, marker="s", color="tab:orange", label="train loss")
    twin.set_ylabel("Train loss")
    axis.set_title(out_path.stem.replace("history_", ""))
    figure.tight_layout()
    figure.savefig(out_path, dpi=140)
    plt.close(figure)


def save_comparison_chart(results: list, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [row["model"] for row in results]
    top1 = [row["accuracy"] for row in results]
    top5 = [row["top5"] for row in results]
    positions = np.arange(len(names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(positions - width / 2, top1, width, label="Top-1")
    axis.bar(positions + width / 2, top5, width, label="Top-5")
    axis.set_xticks(positions, names)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test accuracy")
    axis.set_title("Test recognition rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_path, dpi=140)
    plt.close(figure)


def write_report(results: list, meta: dict, out_path: Path, best_name: str) -> None:
    full = int(meta.get("num_classes", 0)) == 13065 and int(meta.get("max_classes_used", 0) or 0) == 0
    header = "完整 13,065 類測試結果。" if full else (
        f"這次只有 {meta.get('num_classes')} 類，不能當作作業最終辨識率。"
    )
    lines = [
        "# 測試集辨識率",
        "",
        header,
        "",
        f"- 訓練樣本：{meta.get('train_examples')}",
        f"- 測試樣本：{meta.get('test_examples')}",
        f"- 影像大小：{meta.get('img_size')}×{meta.get('img_size')}",
        f"- 每個字的訓練樣本目標：{meta.get('samples_per_class_target', 200)}",
        "- 測試集沒有做旋轉、剪切或縮放。",
        "- 辨識率是最後一個 epoch 的 Top-1 accuracy，測試集沒有拿來選 epoch。",
        "",
        "| 模型 | 架構 | 參數量 | 測試 loss | 辨識率 Top-1 | Top-5 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['model']} | {row['description']} | {row['params']} | {row['loss']:.4f} | "
            f"{row['accuracy']:.4f} | {row['top5']:.4f} |"
        )
    lines.extend(["", f"最佳模型：`{best_name}`", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    configure_stdio()
    if sys.version_info >= (3, 14):
        print("TensorFlow 目前沒有 Python 3.14 的安裝包。請改用 Python 3.9 到 3.13。")
        return 1

    from src.models import MODEL_NAMES, build_model, model_description

    parser = argparse.ArgumentParser(description="Train and compare handwriting models.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--records", type=Path, help="prepare_records.py 產生的資料夾")
    source.add_argument("--data-dir", type=Path, help="小量資料夾。完整 13,065 字請用 --records。")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--work-size", type=int, default=128)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--max-classes", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="從 models/ 與 results/ 裡上次的 epoch 接續。")
    args = parser.parse_args()

    import numpy as np
    import tensorflow as tf

    from src.data import load_labels, load_meta, make_folder_dataset, make_record_dataset

    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    configure_gpu()

    model_dir = Path("models")
    result_dir = Path("results")
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.records:
        meta = load_meta(args.records)
        if int(meta["img_size"]) != args.img_size:
            print(f"TFRecord 的影像是 {meta['img_size']}，與 --img-size {args.img_size} 不同。請沿用準備資料時的大小。")
            return 1
        labels = load_labels(args.records)
        num_classes = len(labels)
        if args.max_classes:
            num_classes = min(num_classes, args.max_classes)
            print(f"警告：只訓練前 {num_classes} 個字。這不是作業最終結果。")
        train_ds = make_record_dataset(args.records, "train", args.batch_size, True, args.seed, args.max_classes)
        test_ds = make_record_dataset(args.records, "test", args.batch_size, False, args.seed, args.max_classes)
        meta = dict(meta)
        meta["num_classes"] = num_classes
        meta["max_classes_used"] = args.max_classes
        meta["samples_per_class_target"] = meta.get("samples_per_class_target", args.samples_per_class)
    else:
        work_size = max(args.work_size, args.img_size)
        train_ds, test_ds, meta = make_folder_dataset(
            args.data_dir,
            args.img_size,
            work_size,
            args.samples_per_class,
            args.batch_size,
            args.seed,
            args.max_classes,
        )
        labels = meta.pop("labels")
        num_classes = int(meta["num_classes"])
        meta["max_classes_used"] = args.max_classes
        meta["samples_per_class_target"] = args.samples_per_class
        (result_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    print(f"類別數 {num_classes}，訓練樣本 {meta['train_examples']}，測試樣本 {meta['test_examples']}")
    if num_classes != 13065:
        print("警告：輸出層不是 13,065。正式繳交請用完整資料集，且不要加 --max-classes。")

    # TFRecord cardinality is unknown, so Keras would stop a later epoch early.
    # Repeat the training set and cut each epoch at the real batch count.
    steps_per_epoch = None
    test_steps = None
    if not args.max_classes:
        steps_per_epoch = math.ceil(int(meta["train_examples"]) / args.batch_size)
        test_steps = math.ceil(int(meta["test_examples"]) / args.batch_size)
        train_ds = train_ds.repeat()
        test_ds = test_ds.repeat()

    results = []
    for name in args.models:
        print(f"\n===== {name} =====")
        csv_path = result_dir / f"history_{name}.csv"
        model_path = model_dir / f"{name}.keras"
        done = completed_epochs(csv_path) if args.resume and model_path.exists() else 0
        if done > 0:
            print(f"接續訓練，已完成 {done} 個 epoch。")
            model = tf.keras.models.load_model(model_path)
            initial_epoch = done
        else:
            if csv_path.exists():
                csv_path.unlink()
            model = build_model(name, num_classes, int(meta["img_size"]), args.learning_rate)
            initial_epoch = 0

        summary_path = result_dir / f"summary_{name}.txt"
        with summary_path.open("w", encoding="utf-8") as handle:
            model.summary(print_fn=lambda line: handle.write(line + "\n"))

        if initial_epoch < args.epochs:
            callbacks = [
                tf.keras.callbacks.CSVLogger(str(csv_path), append=initial_epoch > 0),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=1, min_lr=1e-5, verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="loss", save_best_only=False),
            ]
            model.fit(
                train_ds,
                epochs=args.epochs,
                initial_epoch=initial_epoch,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                verbose=1,
            )
        else:
            print("已達到指定 epoch，直接評估測試集。")

        # Keep a model file even when training was already finished.
        model.save(model_path)
        metrics = model.evaluate(test_ds, steps=test_steps, return_dict=True, verbose=1)
        row = {
            "model": name,
            "description": model_description(name),
            "params": int(model.count_params()),
            "loss": float(metrics["loss"]),
            "accuracy": float(metrics["accuracy"]),
            "top5": float(metrics["top5"]),
        }
        results.append(row)
        print(f"{name} 測試辨識率 Top-1={row['accuracy']:.4f}  Top-5={row['top5']:.4f}")
        if csv_path.exists():
            save_history_plot(csv_path, result_dir / f"history_{name}.png")

    if not results:
        print("沒有模型結果。")
        return 1

    best = max(results, key=lambda row: row["accuracy"])
    shutil.copyfile(model_dir / f"{best['model']}.keras", model_dir / "best_model.keras")
    label_source = args.records / "labels.txt" if args.records else result_dir / "labels.txt"
    shutil.copyfile(label_source, model_dir / "labels.txt")
    (result_dir / "best_model.txt").write_text(best["model"] + "\n", encoding="utf-8")
    (result_dir / "comparison.json").write_text(
        json.dumps({"meta": meta, "results": results, "best": best["model"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(results, meta, result_dir / "comparison.md", best["model"])
    save_comparison_chart(results, result_dir / "comparison.png")
    print(f"\n最佳模型：{best['model']}，測試辨識率 {best['accuracy']:.4f}")
    print("結果已寫入 results/comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
