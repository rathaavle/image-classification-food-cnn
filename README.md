# Image Classification Food CNN

Klasifikasi gambar makanan (Burger, Pizza, Sushi) menggunakan CNN Sequential dengan TensorFlow/Keras.

## Struktur Project

```
image-classification-food-cnn/
├── dataset/
│   ├── train/  burger/ pizza/ sushi/
│   ├── val/    burger/ pizza/ sushi/
│   └── test/   burger/ pizza/ sushi/
├── models/
│   ├── saved_model/     ← Keras SavedModel
│   ├── tflite/          ← model.tflite
│   └── tfjs_model/      ← TensorFlow.js
├── notebooks/
│   └── Template_Submission_Akhir.ipynb
└── src/
    ├── prepare_dataset.py
    ├── preprocess.py
    ├── train.py
    └── inference.py
```

## Setup Dataset

Dataset menggunakan **Food-101** dari Kaggle (kelas: burger, pizza, sushi).
Minimal 1000 gambar per kelas, total ≥ 3000 gambar.

### 1. Install Kaggle CLI

```bash
pip install kaggle
```

### 2. Letakkan API token

Simpan `kaggle.json` di `%USERPROFILE%\.kaggle\kaggle.json`

### 3. Download dataset

```bash
kaggle datasets download -d trolukovich/food-101 -p dataset/raw --unzip
```

### 4. Siapkan dataset (split train/val/test)

```bash
python src/prepare_dataset.py --source dataset/raw/images --target dataset
```

## Training

```bash
cd src
python train.py
```

Training akan:

- Menggunakan augmentasi (rotation, flip, zoom, brightness, dll.)
- Menyimpan checkpoint terbaik berdasarkan `val_accuracy`
- EarlyStopping (patience=7) + ReduceLROnPlateau
- Plot loss & accuracy → `models/training_history.png`
- Menyimpan model ke SavedModel, TFLite, dan TFJS

## Inferensi

```bash
# Menggunakan SavedModel
python src/inference.py path/to/image.jpg

# Menggunakan TFLite
python src/inference.py path/to/image.jpg --tflite
```

## Notebook Submission

Buka `notebooks/Template_Submission_Akhir.ipynb` untuk menjalankan seluruh pipeline
(data loading → preprocessing → training → evaluasi → konversi → inferensi) secara interaktif.

## Arsitektur Model

```
Input (224×224×3)
  → Conv2D(32) + BN + MaxPool
  → Conv2D(64) + BN + MaxPool
  → Conv2D(128) + BN + MaxPool
  → Conv2D(256) + BN + MaxPool
  → Conv2D(512) + BN + MaxPool
  → GlobalAveragePooling2D
  → Dense(512) + Dropout(0.5)
  → Dense(256) + Dropout(0.3)
  → Dense(3, softmax)
```

## Target Akurasi

| Target  | Akurasi |
| ------- | ------- |
| Minimal | ≥ 85%   |
| Aman    | 90–95%  |

## Install Dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn tensorflowjs
```
