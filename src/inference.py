"""
inference.py
Inferensi gambar tunggal menggunakan model yang sudah disimpan.
Mendukung: SavedModel, TFLite, TFJS (via tfjs-node)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# ─── Konstanta ────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
SAVED_MODEL_DIR = BASE_DIR / "models" / "saved_model"
TFLITE_PATH     = BASE_DIR / "models" / "tflite" / "model.tflite"
IMG_SIZE        = (224, 224)
CLASS_NAMES     = ["burger", "pizza", "sushi"]


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    """Load dan preprocess gambar menjadi tensor siap inferensi."""
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0                  # rescale [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # tambah batch dim
    return img_array


# ─── Inferensi SavedModel ─────────────────────────────────────────────────────
def predict_saved_model(image_path: str) -> dict:
    """Prediksi menggunakan SavedModel (Keras .keras / SavedModel format)."""
    model = tf.keras.models.load_model(str(SAVED_MODEL_DIR))
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)
    return _format_result(predictions[0])


# ─── Inferensi TFLite ─────────────────────────────────────────────────────────
def predict_tflite(image_path: str) -> dict:
    """Prediksi menggunakan TFLite interpreter."""
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_array = preprocess_image(image_path).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]["index"])[0]
    return _format_result(predictions)


# ─── Helper ───────────────────────────────────────────────────────────────────
def _format_result(predictions: np.ndarray) -> dict:
    """Format output prediksi menjadi dict yang mudah dibaca."""
    predicted_idx   = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(predictions[predicted_idx])

    result = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {cls: float(prob) for cls, prob in zip(CLASS_NAMES, predictions)},
    }
    return result


def print_result(result: dict, image_path: str = ""):
    """Tampilkan hasil prediksi ke konsol."""
    print("\n" + "=" * 45)
    if image_path:
        print(f"Gambar  : {image_path}")
    print(f"Prediksi: {result['predicted_class'].upper()}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print("\nProbabilitas per kelas:")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"  {cls:<8} {prob*100:5.2f}%  {bar}")
    print("=" * 45)


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image> [--tflite]")
        print("  --tflite  : gunakan TFLite model (default: SavedModel)")
        sys.exit(1)

    image_path = sys.argv[1]
    use_tflite = "--tflite" in sys.argv

    if not os.path.exists(image_path):
        print(f"Error: file tidak ditemukan: {image_path}")
        sys.exit(1)

    if use_tflite:
        print("Menggunakan TFLite model...")
        result = predict_tflite(image_path)
    else:
        print("Menggunakan SavedModel...")
        result = predict_saved_model(image_path)

    print_result(result, image_path)
