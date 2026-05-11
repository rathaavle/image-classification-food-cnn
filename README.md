# Image Classification Food CNN

Klasifikasi gambar makanan (Burger, Pizza, Sushi) menggunakan CNN Sequential dengan TensorFlow/Keras.

## Struktur Project

```
image-classification-food-cnn/
├── dataset/
│   ├── train/   burger/ pizza/ sushi/   (800 gambar/kelas)
│   ├── val/     burger/ pizza/ sushi/   (100 gambar/kelas)
│   └── test/    burger/ pizza/ sushi/   (100 gambar/kelas)
├── models/
│   ├── saved_model/        ← best_checkpoint.keras + model.keras + model.h5
│   ├── tflite/             ← model.tflite
│   ├── tfjs_model/         ← TensorFlow.js
│   └── training_history.png
├── notebooks/
│   └── Submission_Akhir.ipynb
├── src/
│   ├── download_dataset.py
│   ├── prepare_dataset.py
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

Dataset menggunakan **Food-101** via **TensorFlow Datasets** (tidak perlu Kaggle API).
Kelas yang digunakan: `burger` (hamburger), `pizza`, `sushi` — masing-masing 1000 gambar.

### Download & split otomatis

```bash
python src/download_dataset.py
```

Script ini akan:

- Download Food-101 otomatis (~5 GB) via `tensorflow-datasets`
- Memilih 1000 gambar per kelas
- Split ke `train/` (800), `val/` (100), `test/` (100) secara otomatis

> Jika dataset sudah ada secara manual, gunakan `prepare_dataset.py` untuk split dari folder sumber:
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
- Menyimpan model ke format `.keras`, `.tflite`, dan TensorFlow.js

## Konversi Model (Opsional)

Jika ingin mengkonversi ulang dari checkpoint tanpa training ulang:

```bash
python src/convert_model.py
```

Menghasilkan:

- `models/saved_model/model.keras` — Keras 3 native
- `models/tflite/model.tflite` — TFLite quantized
- `models/tfjs_model/` — TensorFlow.js graph model

## Inferensi

```bash
# Menggunakan SavedModel (default)
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

## Dependencies

Versi lengkap ada di `requirements.txt`. Paket utama:

| Package      | Versi  |
| ------------ | ------ |
| tensorflow   | 2.17.0 |
| tf-keras     | 2.17.0 |
| tensorflowjs | 4.10.0 |
| numpy        | 1.26.4 |
| scikit-learn | 1.5.2  |
| matplotlib   | 3.9.2  |
| Pillow       | 10.4.0 |
| jupyter      | 1.1.1  |
