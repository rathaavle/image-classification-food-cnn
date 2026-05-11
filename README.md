# Image Classification Food CNN

Klasifikasi gambar makanan (Burger, Pizza, Sushi) menggunakan CNN Sequential dengan TensorFlow/Keras.

## Struktur Project

```
image-classification-food-cnn/
├── dataset/
│   ├── all/     burger/ pizza/ sushi/   (1000 gambar/kelas — sumber split)
│   ├── train/   burger/ pizza/ sushi/   (800 gambar/kelas  — 80%)
│   ├── val/     burger/ pizza/ sushi/   (100 gambar/kelas  — 10%)
│   └── test/    burger/ pizza/ sushi/   (100 gambar/kelas  — 10%)
├── models/
│   ├── saved_model/
│   │   ├── saved_model/        ← TF SavedModel (saved_model.pb + variables/)
│   │   ├── best_checkpoint.keras
│   │   └── model.h5
│   ├── tflite/
│   │   ├── model.tflite
│   │   └── label.txt           ← nama kelas per baris (burger/pizza/sushi)
│   ├── tfjs_model/             ← TensorFlow.js graph model
│   └── training_history.png
├── notebooks/
│   └── Submission_Akhir.ipynb
├── src/
│   ├── download_dataset.py     ← download + split otomatis via split-folders
│   ├── prepare_dataset.py      ← split manual dari folder sumber lokal
│   ├── preprocess.py
│   ├── train.py
│   ├── convert_model.py
│   └── inference.py
├── requirements.txt
└── .gitignore
```

## Setup & Instalasi

### 1. Buat virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Setup Dataset

Dataset menggunakan **Food-101** via **TensorFlow Datasets**.
Kelas yang digunakan: `burger` (hamburger), `pizza`, `sushi` — 1000 gambar per kelas.

Pembagian dilakukan **secara mandiri** menggunakan library `split-folders`:

```bash
python src/download_dataset.py
```

Alur yang terjadi:

1. Download Food-101 otomatis (~5 GB) via `tensorflow-datasets`
2. Simpan 1000 gambar/kelas ke `dataset/all/<kelas>/`
3. Split secara random dengan seed=42 → `train/` (80%), `val/` (10%), `test/` (10%)

> Jika dataset sudah ada secara lokal, gunakan `prepare_dataset.py`:
>
> ```bash
> python src/prepare_dataset.py --source dataset/raw/images --target dataset
> ```

## Training

```bash
python src/train.py
```

Training akan:

- Menggunakan augmentasi (rotation, flip, zoom, brightness, dll.)
- Menyimpan checkpoint terbaik berdasarkan `val_accuracy` → `models/saved_model/best_checkpoint.keras`
- EarlyStopping (patience=7) + ReduceLROnPlateau (factor=0.3, patience=3)
- Plot loss & accuracy → `models/training_history.png`
- Menyimpan model ke format **TF SavedModel**, **TFLite + label.txt**, dan **TensorFlow.js**

## Konversi Model (Opsional)

Jika ingin mengkonversi ulang dari checkpoint tanpa training ulang:

```bash
python src/convert_model.py
```

Menghasilkan:

- `models/saved_model/saved_model/` — TF SavedModel (`saved_model.pb` + `variables/`)
- `models/tflite/model.tflite` — TFLite quantized
- `models/tflite/label.txt` — nama kelas per baris sesuai urutan indeks
- `models/tfjs_model/` — TensorFlow.js graph model

## Inferensi

```bash
# Menggunakan TF SavedModel (default)
python src/inference.py path/to/image.jpg

# Menggunakan TFLite
python src/inference.py path/to/image.jpg --tflite
```

Output contoh:

```
=============================================
Gambar  : burger_test_0001.jpg
Prediksi: BURGER
Confidence: 97.43%

Probabilitas per kelas:
  burger   97.43%  █████████████████████████████
  pizza     1.82%
  sushi     0.75%
=============================================
```

## Notebook

Buka `notebooks/Submission_Akhir.ipynb` untuk menjalankan seluruh pipeline secara interaktif:
data loading → preprocessing → training → evaluasi → konversi → inferensi.

## Arsitektur Model

```
Input (224×224×3)
  → Conv2D(32)  + BatchNorm + MaxPool
  → Conv2D(64)  + BatchNorm + MaxPool
  → Conv2D(128) + BatchNorm + MaxPool
  → Conv2D(256) + BatchNorm + MaxPool
  → Conv2D(512) + BatchNorm + MaxPool
  → GlobalAveragePooling2D
  → Dense(512) + Dropout(0.5)
  → Dense(256) + Dropout(0.3)
  → Dense(3, softmax)
```

- Optimizer: Adam (lr=1e-3)
- Loss: Categorical Crossentropy
- Input size: 224×224×3
- Output: 3 kelas (burger, pizza, sushi)

## Target Akurasi

| Target  | Akurasi |
| ------- | ------- |
| Minimal | ≥ 85%   |
| Aman    | 90–95%  |

## Dependencies

Versi lengkap ada di `requirements.txt`. Paket utama:

| Package             | Versi  |
| ------------------- | ------ |
| tensorflow          | 2.17.0 |
| tf-keras            | 2.17.0 |
| tensorflowjs        | 4.10.0 |
| tensorflow-datasets | 4.9.6  |
| split-folders       | 0.5.1  |
| numpy               | 1.26.4 |
| scikit-learn        | 1.5.2  |
| matplotlib          | 3.9.2  |
| Pillow              | 10.4.0 |
| jupyter             | 1.1.1  |
