import argparse
import random
import shutil
from pathlib import Path

DEFAULT_CLASSES = ["burger", "pizza", "sushi"]
DEFAULT_SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Kaggle Food-101 dataset for this project.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset/raw/images"),
        help="Lokasi gambar Food-101 yang diunduh dan diekstrak.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("dataset"),
        help="Folder output target dataset (train/val/test).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Daftar kelas yang akan diproses.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed RNG untuk membagi data secara acak.",
    )
    return parser.parse_args()


def split_files(file_paths, ratios):
    random.shuffle(file_paths)
    total = len(file_paths)
    train_end = int(total * ratios["train"])
    val_end = train_end + int(total * ratios["val"])
    return {
        "train": file_paths[:train_end],
        "val": file_paths[train_end:val_end],
        "test": file_paths[val_end:],
    }


def copy_files(file_paths, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src_path in file_paths:
        dest_path = dest_dir / src_path.name
        shutil.copy2(src_path, dest_path)


def prepare_dataset(source_dir: Path, target_dir: Path, classes, seed: int):
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source path tidak ditemukan: {source_dir}. Pastikan dataset Food-101 sudah diunduh dan diekstrak."
        )

    random.seed(seed)

    for cls in classes:
        class_source = source_dir / cls
        if not class_source.exists() or not class_source.is_dir():
            print(f"Peringatan: kelas tidak ditemukan, melewatkan {cls}: {class_source}")
            continue

        image_paths = [p for p in class_source.iterdir() if p.is_file()]
        if not image_paths:
            print(f"Peringatan: tidak ada file gambar di {class_source}")
            continue

        splits = split_files(image_paths, DEFAULT_SPLIT)
        for split_name, paths in splits.items():
            dest_class_dir = target_dir / split_name / cls
            copy_files(paths, dest_class_dir)
            print(f"{split_name}: {len(paths)} gambar disalin ke {dest_class_dir}")


def main():
    args = parse_args()
    print("Memulai persiapan dataset...")
    print(f"Sumber: {args.source}")
    print(f"Target: {args.target}")
    print(f"Kelas: {args.classes}")

    prepare_dataset(args.source, args.target, args.classes, args.seed)
    print("Selesai. Cek folder dataset/train, dataset/val, dataset/test.")


if __name__ == "__main__":
    main()
