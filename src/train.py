import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if "KERAS_HOME" not in os.environ and (ROOT / ".keras").exists():
    os.environ["KERAS_HOME"] = str((ROOT / ".keras").resolve())
if "MPLCONFIGDIR" not in os.environ and (ROOT / ".keras" / "matplotlib").exists():
    os.environ["MPLCONFIGDIR"] = str((ROOT / ".keras" / "matplotlib").resolve())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import make_dataset, CLASSES
from models import build_model

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "isl_model.keras"
WEIGHTS_PATH = MODEL_DIR / "model.weights.h5"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def train(epochs=3, batch_size=64, learning_rate=5e-5):
    tf.keras.utils.set_random_seed(42)

    print("Loading Training and Validation datasets...")
    train_ds = make_dataset("Training", training=True, batch_size=batch_size)
    val_ds = make_dataset("Validation", training=False, batch_size=batch_size)

    # Build model unfreezing top layers (from block 14 onwards)
    FINE_TUNE_AT = 120
    model = build_model(trainable_base=True, fine_tune_at=FINE_TUNE_AT)

    # Warm-start from baseline weights if available
    if WEIGHTS_PATH.exists():
        print(f"Loading baseline weights from: {WEIGHTS_PATH}")
        model.load_weights(str(WEIGHTS_PATH))
        print("Pre-trained baseline weights loaded successfully.")
    else:
        print("Training from base MobileNetV2 ImageNet features.")

    # Target class weighting to penalize errors on historically confused letters (S, Z, V, Q, X)
    class_weights = {i: 1.0 for i in range(len(CLASSES))}
    class_weights[CLASSES.index("S")] = 3.0
    class_weights[CLASSES.index("Z")] = 3.0
    class_weights[CLASSES.index("V")] = 2.0
    class_weights[CLASSES.index("Q")] = 2.0
    class_weights[CLASSES.index("X")] = 2.0
    class_weights[CLASSES.index("I")] = 1.5

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"\nStarting fine-tuning for {epochs} epochs (Batch size={batch_size}, LR={learning_rate})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Save model and update weights for cross-version compatibility
    model.save(MODEL_PATH)
    try:
        model.save_weights(str(WEIGHTS_PATH))
    except Exception as e:
        print(f"save_weights note: {e}")

    print(f"\nTraining complete. Model saved to: {MODEL_PATH}")

    # Plot curves
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Train Loss", marker="o")
        plt.plot(history.history["val_loss"], label="Val Loss", marker="s")
        plt.title("Fine-tuning Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Train Acc", marker="o")
        plt.plot(history.history["val_accuracy"], label="Val Acc", marker="s")
        plt.title("Fine-tuning Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plot_path = REPORTS_DIR / "improved_training_curves.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Training curves saved to: {plot_path}")

        hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(REPORTS_DIR / "improved_training_history.json", "w") as f:
            json.dump(hist_dict, f, indent=2)
    except Exception as e:
        print(f"History plot warning: {e}")


if __name__ == "__main__":
    train(epochs=3, batch_size=64, learning_rate=5e-5)
