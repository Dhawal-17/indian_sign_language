import os
import json
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Dataset"
INDEX_FILE = ROOT / "reports" / "dataset_index.json"

if "KERAS_HOME" not in os.environ and (ROOT / ".keras").exists():
    os.environ["KERAS_HOME"] = str((ROOT / ".keras").resolve())
if "MPLCONFIGDIR" not in os.environ and (ROOT / ".keras" / "matplotlib").exists():
    os.environ["MPLCONFIGDIR"] = str((ROOT / ".keras" / "matplotlib").resolve())

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def get_valid_images(split):
    # Fast path: use indexed records if available
    if INDEX_FILE.exists():
        try:
            with open(INDEX_FILE, "r") as f:
                index_data = json.load(f)
            if split in index_data:
                records = [
                    (str((ROOT / rel_path).resolve()), label)
                    for rel_path, label in index_data[split]
                ]
                print(f"{split}: {len(records)} valid images (loaded from index)")
                return records
        except Exception:
            pass

    records = []
    split_dir = DATASET_DIR / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Folder not found: {split_dir}")

    for label, letter in enumerate(CLASSES):
        class_dir = split_dir / letter
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for path in class_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            try:
                with Image.open(path) as image:
                    image.verify()
                records.append((str(path), label))
            except (UnidentifiedImageError, OSError, ValueError):
                print(f"Skipping invalid image: {path}")

    if not records:
        raise RuntimeError(f"No valid images found in {split_dir}")

    print(f"{split}: {len(records)} valid images")
    return records


def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def make_dataset(split, training=False, batch_size=BATCH_SIZE):
    records = get_valid_images(split)
    paths, labels = zip(*records)

    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    if training:
        ds = ds.shuffle(len(records), reshuffle_each_iteration=True)

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.04, 0.04),
        ])
        ds = ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    for split in ["Training", "Validation", "Testing"]:
        ds = make_dataset(split, training=(split == "Training"))
        print(f"{split} dataset ready.")
