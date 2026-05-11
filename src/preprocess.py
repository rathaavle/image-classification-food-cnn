"""
preprocess.py
Modul preprocessing dan augmentasi data untuk klasifikasi gambar makanan.
"""

import tensorflow as tf

# Konstanta
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def build_train_datagen():
    """ImageDataGenerator dengan augmentasi untuk data training."""
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )


def build_val_test_datagen():
    """ImageDataGenerator tanpa augmentasi untuk validasi dan test."""
    return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)


def get_train_generator(train_dir: str, batch_size: int = BATCH_SIZE):
    datagen = build_train_datagen()
    return datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )


def get_val_generator(val_dir: str, batch_size: int = BATCH_SIZE):
    datagen = build_val_test_datagen()
    return datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )


def get_test_generator(test_dir: str, batch_size: int = BATCH_SIZE):
    datagen = build_val_test_datagen()
    return datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
