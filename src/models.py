import os
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
if "KERAS_HOME" not in os.environ and (ROOT / ".keras").exists():
    os.environ["KERAS_HOME"] = str((ROOT / ".keras").resolve())
if "MPLCONFIGDIR" not in os.environ and (ROOT / ".keras" / "matplotlib").exists():
    os.environ["MPLCONFIGDIR"] = str((ROOT / ".keras" / "matplotlib").resolve())

IMG_SIZE = 224
NUM_CLASSES = 26
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_MODEL_PATH = ROOT / "models" / "isl_model.keras"
WEIGHTS_PATH = ROOT / "models" / "model.weights.h5"


def build_model(trainable_base=False, fine_tune_at=None):
    """
    Builds the MobileNetV2-based ISL alphabet recognition model.
    Preserves exact architecture and layer naming for full compatibility.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    if not trainable_base:
        base.trainable = False
    else:
        base.trainable = True
        if fine_tune_at is not None:
            for layer in base.layers[:fine_tune_at]:
                layer.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        dtype="float32"
    )(x)

    return tf.keras.Model(inputs, outputs)


def load_isl_model(model_path=None):
    """
    Loads the trained ISL model with automatic fallback handling across
    different Keras 3 versions for robust execution.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # First attempt native load_model
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Native load_model encountered version config difference ({e}).")
        print("Falling back to architecture reconstruction and weights loading...")

    # Fallback: construct architecture and load weights
    model = build_model(trainable_base=False)
    try:
        model.load_weights(model_path)
        return model
    except Exception:
        weights_h5 = model_path.parent / "model.weights.h5"
        if not weights_h5.exists():
            import zipfile
            if zipfile.is_zipfile(model_path):
                with zipfile.ZipFile(model_path) as z:
                    z.extract("model.weights.h5", model_path.parent)
        if weights_h5.exists():
            model.load_weights(str(weights_h5))
            return model
        raise RuntimeError(f"Could not load weights from {model_path}")


if __name__ == "__main__":
    model = build_model()
    model.summary()
