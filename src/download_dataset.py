"""
download_dataset.py
Download subset Food-101 (burger, pizza, sushi) via TensorFlow Datasets,
lalu simpan semua gambar ke folder `dataset/all/<kelas>/` dan split secara
mandiri menggunakan split-folders (80% train / 10% val / 10% test).

Jalankan: python src/download_dataset.py
"""

import os
import sys
import shutil
from pathlib import Path

# ─── Install dependensi jika belum ada ───────────────────────────────────────
def _ensure_package(import_name: str, pip_name: str, version: str = ""):
    try:
        __import__(import_name)
    except ImportError:
        pkg = f"{pip_name}=={version}" if version else pip_name
        print(f"Menginstall {pkg}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

_ensure_package("tensorflow_datasets", "tensorflow-datasets", "4.9.6")
_ensure_package("splitfolders", "split-folders")

import tensorflow_datasets as tfds
import splitfolders
import tensorflow as tf
from PIL import Image

# ─── Konfigurasi ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
ALL_DIR     = DATASET_DIR / "all"          # folder sumber sebelum di-split

# Mapping nama kelas Food-101 → nama folder kita
CLASS_MAP = {
    "hamburger": "burger",   # Food-101 pakai "hamburger"
    "pizza":     "pizza",
    "sushi":     "sushi",
}

TOTAL_PER_CLASS = 1000   # ambil 1000 gambar per kelas dari Food-101
SEED            = 42
SPLIT_RATIO     = (0.8, 0.1, 0.1)   # train / val / test


def save_image(image_tensor, dest_path: Path):
    """Simpan tensor gambar ke file JPEG."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img_array = image_tensor.numpy()
    img = Image.fromarray(img_array.astype("uint8")).convert("RGB")
    img.save(str(dest_path), "JPEG", quality=90)


def download_to_all_folder():
    """Download Food-101 dan simpan semua gambar ke dataset/all/<kelas>/."""
    print("=" * 60)
    print("Download Food-101 via TensorFlow Datasets")
    print("Kelas: burger (hamburger), pizza, sushi")
    print("=" * 60)
    print()
    print("Memuat dataset Food-101 (download otomatis jika belum ada)...")
    print("Ukuran download: ~5 GB — harap tunggu...\n")

    # Gabungkan split train + validation dari TFDS agar kita bisa split sendiri
    ds_train, ds_val = tfds.load(
        "food101",
        split=["train", "validation"],
        as_supervised=True,
        shuffle_files=True,
    )
    ds_all = ds_train.concatenate(ds_val)

    info        = tfds.builder("food101").info
    label_names = info.features["label"].names
    print(f"Total kelas Food-101: {len(label_names)}")

    # Cari index kelas yang dibutuhkan
    target_indices = {}
    for food101_name, our_name in CLASS_MAP.items():
        if food101_name in label_names:
            idx = label_names.index(food101_name)
            target_indices[idx] = our_name
            print(f"  '{food101_name}' → index {idx} → folder '{our_name}'")
        else:
            print(f"  PERINGATAN: '{food101_name}' tidak ditemukan!")
    print()

    # Kumpulkan gambar per kelas
    class_images: dict[str, list] = {name: [] for name in CLASS_MAP.values()}
    print("Mengumpulkan gambar per kelas...")
    for image, label in ds_all:
        label_int = int(label.numpy())
        if label_int in target_indices:
            our_name = target_indices[label_int]
            if len(class_images[our_name]) < TOTAL_PER_CLASS:
                class_images[our_name].append(image)
        if all(len(v) >= TOTAL_PER_CLASS for v in class_images.values()):
            break

    for name, imgs in class_images.items():
        print(f"  {name}: {len(imgs)} gambar terkumpul")
    print()

    # Simpan ke dataset/all/<kelas>/
    print("Menyimpan gambar ke dataset/all/ ...")
    for cls_name, images in class_images.items():
        cls_dir = ALL_DIR / cls_name
        # Bersihkan folder lama jika ada
        if cls_dir.exists():
            shutil.rmtree(cls_dir)
        cls_dir.mkdir(parents=True)
        for i, img_tensor in enumerate(images):
            dest_path = cls_dir / f"{cls_name}_{i:04d}.jpg"
            save_image(img_tensor, dest_path)
        print(f"  all/{cls_name}: {len(images)} gambar disimpan")
    print()


def split_dataset():
    """Split dataset/all/ → train / val / test menggunakan split-folders."""
    print("Membagi dataset secara mandiri dengan split-folders ...")
    print(f"  Rasio: train={SPLIT_RATIO[0]:.0%}  val={SPLIT_RATIO[1]:.0%}  test={SPLIT_RATIO[2]:.0%}")
    print(f"  Seed : {SEED}\n")

    # Hapus folder split lama agar tidak tercampur
    for split in ["train", "val", "test"]:
        split_dir = DATASET_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    # split-folders akan membuat dataset/train, dataset/val, dataset/test
    splitfolders.ratio(
        str(ALL_DIR),
        output=str(DATASET_DIR),
        seed=SEED,
        ratio=SPLIT_RATIO,
        group_prefix=None,
        move=False,
    )
    print("Pembagian selesai.\n")


def verify():
    """Tampilkan ringkasan jumlah gambar per split per kelas."""
    print("=" * 60)
    print("Verifikasi dataset:")
    total = 0
    for split in ["train", "val", "test"]:
        for cls in sorted(CLASS_MAP.values()):
            d = DATASET_DIR / split / cls
            count = len(list(d.glob("*.jpg"))) if d.exists() else 0
            total += count
            print(f"  dataset/{split}/{cls}: {count} gambar")
        print()
    print(f"Total gambar: {total}")
    print("=" * 60)


if __name__ == "__main__":
    download_to_all_folder()
    split_dataset()
    verify()
