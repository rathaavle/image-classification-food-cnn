"""
download_dataset.py
Download subset Food-101 (burger, pizza, sushi) langsung via TensorFlow Datasets.
Tidak perlu Kaggle API atau setup manual apapun.

Jalankan: python src/download_dataset.py
"""

import os
import sys
import shutil
from pathlib import Path

# ─── Install tensorflow-datasets jika belum ada ──────────────────────────────
try:
    import tensorflow_datasets as tfds
except ImportError:
    print("Menginstall tensorflow-datasets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-datasets==4.9.6", "--quiet"])
    import tensorflow_datasets as tfds

import tensorflow as tf
from PIL import Image
import numpy as np

# ─── Konfigurasi ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR     = BASE_DIR / "dataset" / "raw_tfds"

# Mapping nama kelas Food-101 → nama folder kita
CLASS_MAP = {
    "hamburger": "burger",   # Food-101 pakai "hamburger"
    "pizza":     "pizza",
    "sushi":     "sushi",
}

# Target jumlah gambar per kelas (Food-101 punya 1000/kelas)
TARGET_PER_CLASS = {
    "train": 800,
    "val":   100,
    "test":  100,
}

TOTAL_PER_CLASS = sum(TARGET_PER_CLASS.values())  # 1000


def save_image(image_tensor, dest_path: Path):
    """Simpan tensor gambar ke file JPEG."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img_array = image_tensor.numpy()
    img = Image.fromarray(img_array.astype("uint8"))
    img = img.convert("RGB")
    img.save(str(dest_path), "JPEG", quality=90)


def download_and_split():
    print("=" * 60)
    print("Download Food-101 via TensorFlow Datasets")
    print("Kelas: burger (hamburger), pizza, sushi")
    print("=" * 60)
    print()

    # Load Food-101 — akan download otomatis (~5GB) ke cache TFDS
    # Gunakan split train saja karena kita akan split manual
    print("Memuat dataset Food-101 (download otomatis jika belum ada)...")
    print("Ukuran download: ~5GB — harap tunggu...\n")

    # Ambil semua data (train+validation) untuk kita split sendiri
    ds_train, ds_val = tfds.load(
        "food101",
        split=["train", "validation"],
        as_supervised=True,
        shuffle_files=True,
    )

    # Gabungkan semua data
    ds_all = ds_train.concatenate(ds_val)

    # Dapatkan info label
    info = tfds.builder("food101").info
    label_names = info.features["label"].names
    print(f"Total kelas Food-101: {len(label_names)}")

    # Cari index untuk kelas yang kita butuhkan
    target_indices = {}
    for food101_name, our_name in CLASS_MAP.items():
        if food101_name in label_names:
            idx = label_names.index(food101_name)
            target_indices[idx] = our_name
            print(f"  '{food101_name}' → index {idx} → folder '{our_name}'")
        else:
            print(f"  PERINGATAN: '{food101_name}' tidak ditemukan di Food-101!")

    print()

    # Kumpulkan gambar per kelas
    class_images = {name: [] for name in CLASS_MAP.values()}
    print("Mengumpulkan gambar per kelas...")

    for image, label in ds_all:
        label_int = int(label.numpy())
        if label_int in target_indices:
            our_name = target_indices[label_int]
            if len(class_images[our_name]) < TOTAL_PER_CLASS:
                class_images[our_name].append(image)

        # Cek apakah semua kelas sudah terpenuhi
        if all(len(imgs) >= TOTAL_PER_CLASS for imgs in class_images.values()):
            break

    # Laporan
    for name, imgs in class_images.items():
        print(f"  {name}: {len(imgs)} gambar terkumpul")

    print()
    print("Menyimpan gambar ke folder dataset...")

    # Simpan ke train/val/test
    for cls_name, images in class_images.items():
        counts = TARGET_PER_CLASS
        splits = {
            "train": images[:counts["train"]],
            "val":   images[counts["train"]:counts["train"] + counts["val"]],
            "test":  images[counts["train"] + counts["val"]:],
        }

        for split_name, split_images in splits.items():
            dest_dir = DATASET_DIR / split_name / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Hapus file lama jika ada
            for f in dest_dir.glob("*.jpg"):
                f.unlink()

            for i, img_tensor in enumerate(split_images):
                dest_path = dest_dir / f"{cls_name}_{split_name}_{i:04d}.jpg"
                save_image(img_tensor, dest_path)

            print(f"  {split_name}/{cls_name}: {len(split_images)} gambar disimpan")

    print()
    print("=" * 60)
    print("Dataset siap!")
    print()

    # Verifikasi akhir
    total = 0
    for split in ["train", "val", "test"]:
        for cls in CLASS_MAP.values():
            d = DATASET_DIR / split / cls
            count = len(list(d.glob("*.jpg")))
            total += count
            print(f"  dataset/{split}/{cls}: {count} gambar")
        print()

    print(f"Total gambar: {total}")
    print("=" * 60)


if __name__ == "__main__":
    download_and_split()
