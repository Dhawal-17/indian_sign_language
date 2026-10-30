import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if "KERAS_HOME" not in os.environ and (ROOT / ".keras").exists():
    os.environ["KERAS_HOME"] = str((ROOT / ".keras").resolve())
if "MPLCONFIGDIR" not in os.environ and (ROOT / ".keras" / "matplotlib").exists():
    os.environ["MPLCONFIGDIR"] = str((ROOT / ".keras" / "matplotlib").resolve())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data import make_dataset, CLASSES
from models import load_isl_model

MODEL_PATH = ROOT / "models" / "isl_model.keras"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(model_path=MODEL_PATH):
    print(f"Loading model from: {model_path}")
    model = load_isl_model(model_path)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Loading held-out Testing dataset...")
    test_ds = make_dataset("Testing", training=False, batch_size=32)

    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\n==========================================")
    print(f"Test Accuracy: {accuracy * 100:.2f}% | Test Loss: {loss:.4f}")
    print(f"==========================================\n")

    all_preds = []
    all_labels = []

    print("Evaluating predictions across test batches...")
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report_dict = classification_report(
        all_labels, all_preds, target_names=CLASSES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    per_letter = {}
    print(f"{'Letter':<8}{'Accuracy':<12}{'Correct/Total':<16}{'Precision':<12}{'Recall':<10}{'F1-Score':<10}")
    print("-" * 68)
    for i, letter in enumerate(CLASSES):
        idx = (all_labels == i)
        total_i = int(np.sum(idx))
        correct_i = int(np.sum(all_preds[idx] == i))
        acc_i = (correct_i / total_i * 100.0) if total_i > 0 else 0.0
        prec_i = report_dict[letter]["precision"] * 100.0
        rec_i = report_dict[letter]["recall"] * 100.0
        f1_i = report_dict[letter]["f1-score"] * 100.0
        per_letter[letter] = {
            "total": total_i,
            "correct": correct_i,
            "accuracy": acc_i,
            "precision": prec_i,
            "recall": rec_i,
            "f1": f1_i
        }
        print(f"{letter:<8}{acc_i:>6.2f}%     {correct_i:>3d}/{total_i:>3d}          {prec_i:>6.1f}%     {rec_i:>6.1f}%    {f1_i:>6.1f}%")

    print("-" * 68)
    print(f"Key Focus Letters: S = {per_letter['S']['accuracy']:.2f}% | Z = {per_letter['Z']['accuracy']:.2f}%")

    # Plot Confusion Matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES
    )
    plt.title(f"Improved ISL Model Confusion Matrix (Test Accuracy: {accuracy*100:.2f}%)")
    plt.xlabel("Predicted Letter")
    plt.ylabel("True Letter")
    plt.tight_layout()
    cm_plot_path = REPORTS_DIR / "improved_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_plot_path}")

    # Save Improved Results JSON
    results = {
        "overall_accuracy": float(accuracy),
        "loss": float(loss),
        "per_letter": per_letter,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }
    results_path = REPORTS_DIR / "improved_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Before vs After Comparison
    baseline_path = REPORTS_DIR / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path, "r") as f:
            base_data = json.load(f)

        print("\n========================================================")
        print("           BEFORE vs AFTER COMPARISON ON TEST SET       ")
        print("========================================================")
        print(f"{'Metric / Letter':<18}{'Baseline':<14}{'Improved':<14}{'Change':<10}")
        print("-" * 56)
        base_acc = base_data["overall_accuracy"] * 100.0
        imp_acc = accuracy * 100.0
        diff_acc = imp_acc - base_acc
        print(f"{'Overall Accuracy':<18}{base_acc:>6.2f}%        {imp_acc:>6.2f}%        {diff_acc:>+6.2f}%")
        print("-" * 56)

        for letter in CLASSES:
            b_acc = base_data["per_letter"][letter]["accuracy"]
            i_acc = per_letter[letter]["accuracy"]
            diff = i_acc - b_acc
            marker = " *" if letter in ["S", "Z"] else ""
            print(f"{letter + marker:<18}{b_acc:>6.2f}%        {i_acc:>6.2f}%        {diff:>+6.2f}%")
        print("-" * 56)
        print("(* indicates focal letters S and Z)")


if __name__ == "__main__":
    evaluate()
