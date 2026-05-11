"""
convert_model.py
Konversi model dari best_checkpoint.keras ke:
  1. model.keras  (Keras 3 native)
  2. model.tflite (TFLite quantized)
  3. tfjs_model/  (TensorFlow.js)

Jalankan: python src/convert_model.py
"""

import os
import sys
import numpy as np
import tensorflow as tf

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_model")
TFLITE_DIR      = os.path.join(BASE_DIR, "models", "tflite")
TFJS_DIR        = os.path.join(BASE_DIR, "models", "tfjs_model")
CHECKPOINT_PATH = os.path.join(SAVED_MODEL_DIR, "best_checkpoint.keras")
KERAS_OUT_PATH  = os.path.join(SAVED_MODEL_DIR, "model.keras")
H5_OUT_PATH     = os.path.join(SAVED_MODEL_DIR, "model.h5")
TFLITE_PATH     = os.path.join(TFLITE_DIR, "model.tflite")

IMG_SIZE    = (224, 224)
NUM_CLASSES = 3

os.makedirs(TFLITE_DIR, exist_ok=True)
os.makedirs(TFJS_DIR, exist_ok=True)

print("=" * 55)
print("Konversi Model Food CNN")
print("=" * 55)

# ── Load model dari checkpoint ────────────────────────────────
print(f"\n[1/4] Load model dari: {CHECKPOINT_PATH}")
model = tf.keras.models.load_model(CHECKPOINT_PATH)
print("      Model loaded OK")
print(f"      Input shape : {model.input_shape}")
print(f"      Output shape: {model.output_shape}")

# ── Simpan .keras (Keras 3 native) ───────────────────────────
print(f"\n[2/4] Simpan .keras → {KERAS_OUT_PATH}")
model.save(KERAS_OUT_PATH)
print(f"      OK  ({os.path.getsize(KERAS_OUT_PATH)/1024:.0f} KB)")

# ── Konversi ke TFLite ────────────────────────────────────────
print(f"\n[3/4] Konversi ke TFLite → {TFLITE_PATH}")
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
)
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func], model
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"      OK  ({os.path.getsize(TFLITE_PATH)/1024:.0f} KB)")

# Verifikasi TFLite
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()
dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
interp.set_tensor(inp[0]["index"], dummy)
interp.invoke()
pred = interp.get_tensor(out[0]["index"])
print(f"      Verifikasi TFLite: output shape {pred.shape} ✓")

# ── Konversi ke TFJS ──────────────────────────────────────────
print(f"\n[4/4] Konversi ke TFJS → {TFJS_DIR}")

# Simpan dulu ke H5 via tf_keras (legacy) agar kompatibel dengan tensorflowjs
import tf_keras

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
    tf_keras.layers.Dense(NUM_CLASSES, activation="softmax"),
], name="food_cnn_h5")

tf_keras_model.set_weights(model.get_weights())
tf_keras_model.save(H5_OUT_PATH)
print(f"      H5 tersimpan: {H5_OUT_PATH} ({os.path.getsize(H5_OUT_PATH)/1024:.0f} KB)")

import subprocess
venv_dir = os.path.dirname(sys.executable)
converter_exe = os.path.join(venv_dir, "tensorflowjs_converter")

# Set TF_USE_LEGACY_KERAS=1 agar tensorflowjs_converter pakai tf_keras bukan Keras 3
env = os.environ.copy()
env["TF_USE_LEGACY_KERAS"] = "1"

result = subprocess.run(
    [converter_exe,
     "--input_format=keras",
     "--output_format=tfjs_graph_model",
     H5_OUT_PATH,
     TFJS_DIR],
    capture_output=True, text=True,
    env=env
)

if result.returncode == 0:
    files = os.listdir(TFJS_DIR)
    print(f"      OK  → {files}")
else:
    print("      GAGAL:")
    print(result.stderr[-400:])
    sys.exit(1)

# ── Ringkasan ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("SELESAI — Semua model tersimpan:")
print(f"  .keras  : {KERAS_OUT_PATH}")
print(f"  .tflite : {TFLITE_PATH}")
print(f"  tfjs    : {TFJS_DIR}/")
print("=" * 55)
