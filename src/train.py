"""
train.py
Script training model CNN Sequential untuk klasifikasi makanan (burger, pizza, sushi).
Target akurasi: ≥ 85% (target aman 90–95%)
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from preprocess import (
    IMG_SIZE,
    BATCH_SIZE,
    get_train_generator,
    get_val_generator,
    get_test_generator,
)

# ─── Path ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR   = os.path.join(BASE_DIR, "dataset", "val")
TEST_DIR  = os.path.join(BASE_DIR, "dataset", "test")

SAVED_MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_model")
TFLITE_DIR      = os.path.join(BASE_DIR, "models", "tflite")
TFJS_DIR        = os.path.join(BASE_DIR, "models", "tfjs_model")

NUM_CLASSES = 3
EPOCHS      = 30


# ─── Model ───────────────────────────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    CNN Sequential dengan Conv2D + MaxPooling + BatchNorm + Dropout.
    Input: (224, 224, 3)
    """
    model = models.Sequential(
        [
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                          input_shape=(*IMG_SIZE, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 4
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 5
            layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Classifier head
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="food_cnn",
    )
    return model


# ─── Callbacks ───────────────────────────────────────────────────────────────
def build_callbacks(checkpoint_path: str) -> list:
    """
    Kumpulan callback:
    - ModelCheckpoint  : simpan bobot terbaik berdasarkan val_accuracy
    - EarlyStopping    : hentikan training jika tidak ada peningkatan
    - ReduceLROnPlateau: turunkan learning rate saat plateau
    - TensorBoard      : logging (opsional)
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    )

    return [checkpoint, early_stop, reduce_lr]


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_history(history, save_path: str = None):
    """Plot loss & accuracy training vs validasi."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],    label="Train Accuracy",  color="royalblue")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",   color="tomato")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss",  color="royalblue")
    axes[1].plot(history.history["val_loss"], label="Val Loss",    color="tomato")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot disimpan ke: {save_path}")
    plt.show()


# ─── Save & Convert ──────────────────────────────────────────────────────────
def save_keras_model(model, saved_model_dir: str) -> str:
    """Simpan model ke format .keras (Keras 3 native)."""
    os.makedirs(saved_model_dir, exist_ok=True)
    keras_path = os.path.join(saved_model_dir, "model.keras")
    model.save(keras_path)
    print(f"Model .keras disimpan ke: {keras_path}")
    return keras_path


def save_h5_via_tf_keras(model, saved_model_dir: str, num_classes: int) -> str:
    """
    Simpan model ke .h5 menggunakan tf_keras (legacy Keras 2).
    Diperlukan agar tensorflowjs_converter bisa membaca model.
    """
    import tf_keras
    os.makedirs(saved_model_dir, exist_ok=True)
    h5_path = os.path.join(saved_model_dir, "model.h5")

    # Rebuild arsitektur yang sama dengan tf_keras
    tf_keras_model = tf_keras.models.Sequential([
        tf_keras.layers.Conv2D(32, (3,3), activation="relu", padding="same",
                               input_shape=(*IMG_SIZE, 3)),
        tf_keras.layers.BatchNormalization(),
        tf_keras.layers.MaxPooling2D((2,2)),
        tf_keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf_keras.layers.BatchNormalization(),
        tf_keras.layers.MaxPooling2D((2,2)),
        tf_keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        tf_keras.layers.BatchNormalization(),
        tf_keras.layers.MaxPooling2D((2,2)),
        tf_keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        tf_keras.layers.BatchNormalization(),
        tf_keras.layers.MaxPooling2D((2,2)),
        tf_keras.layers.Conv2D(512, (3,3), activation="relu", padding="same"),
        tf_keras.layers.BatchNormalization(),
        tf_keras.layers.MaxPooling2D((2,2)),
        tf_keras.layers.GlobalAveragePooling2D(),
        tf_keras.layers.Dense(512, activation="relu"),
        tf_keras.layers.Dropout(0.5),
        tf_keras.layers.Dense(256, activation="relu"),
        tf_keras.layers.Dropout(0.3),
        tf_keras.layers.Dense(num_classes, activation="softmax"),
    ], name="food_cnn_h5")

    # Transfer bobot dari model Keras 3
    tf_keras_model.set_weights(model.get_weights())
    tf_keras_model.save(h5_path)
    print(f"Model .h5 disimpan ke: {h5_path}")
    return h5_path


def convert_to_tflite(model, tflite_dir: str):
    """Konversi ke TFLite via concrete function (kompatibel Keras 3 + TF 2.17)."""
    os.makedirs(tflite_dir, exist_ok=True)
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = os.path.join(tflite_dir, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model disimpan ke: {tflite_path}")
    print(f"Ukuran TFLite: {os.path.getsize(tflite_path) / 1024:.1f} KB")


def convert_to_tfjs(h5_path: str, tfjs_dir: str):
    """Konversi .h5 ke TensorFlow.js menggunakan tensorflowjs_converter."""
    import subprocess
    import sys
    os.makedirs(tfjs_dir, exist_ok=True)
    venv_dir = os.path.dirname(sys.executable)
    converter_path = os.path.join(venv_dir, "tensorflowjs_converter")
    cmd = [
        converter_path,
        "--input_format=keras",
        "--output_format=tfjs_graph_model",
        h5_path,
        tfjs_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"TFJS model disimpan ke: {tfjs_dir}")
        print(f"File TFJS: {os.listdir(tfjs_dir)}")
    else:
        print("Gagal konversi TFJS:")
        print(result.stderr[-400:])


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Food CNN Training — burger | pizza | sushi")
    print("=" * 60)

    # 1. Data generators
    train_gen = get_train_generator(TRAIN_DIR, BATCH_SIZE)
    val_gen   = get_val_generator(VAL_DIR,   BATCH_SIZE)
    test_gen  = get_test_generator(TEST_DIR,  BATCH_SIZE)

    print(f"\nKelas: {train_gen.class_indices}")
    print(f"Train samples : {train_gen.samples}")
    print(f"Val samples   : {val_gen.samples}")
    print(f"Test samples  : {test_gen.samples}\n")

    # 2. Build model
    model = build_model(NUM_CLASSES)
    model.summary()

    # 3. Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 4. Callbacks
    checkpoint_path = os.path.join(SAVED_MODEL_DIR, "best_checkpoint.keras")
    cb_list = build_callbacks(checkpoint_path)

    # 5. Train
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=cb_list,
    )

    # 6. Evaluate on test set
    print("\n--- Evaluasi pada Test Set ---")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # 7. Plot
    plot_path = os.path.join(BASE_DIR, "models", "training_history.png")
    plot_history(history, save_path=plot_path)

    # 8. Save models
    save_keras_model(model, SAVED_MODEL_DIR)
    h5_path = save_h5_via_tf_keras(model, SAVED_MODEL_DIR, NUM_CLASSES)
    convert_to_tflite(model, TFLITE_DIR)
    convert_to_tfjs(h5_path, TFJS_DIR)

    print("\nTraining selesai!")


if __name__ == "__main__":
    main()
