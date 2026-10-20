"""
DoodleNet: Real-Time Sketch Recognition - CNN Training Pipeline

This script downloads real human sketch bitmaps from Google's Quick, Draw! dataset
using fast HTTP range requests (cached in ./data/), trains a high-accuracy Convolutional
Neural Network (CNN), and outputs model.h5 along with classes.json.
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow / Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.utils import to_categorical
except ImportError:
    print("\n[ERROR] TensorFlow is not installed in the current environment.")
    print("Please install dependencies: pip install -r requirements.txt\n")
    sys.exit(1)

# Target 10 Classes for DoodleNet
CLASSES = [
    "apple",
    "bus",
    "cat",
    "dog",
    "fish",
    "flower",
    "house",
    "key",
    "pencil",
    "star"
]

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download_class_samples(class_name, n_samples=4000, data_dir=DATA_DIR):
    """
    Downloads real human sketches from Quick, Draw! using HTTP Range requests.
    Saves the extracted slice to ./data/{class_name}.npy for fast local reuse.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, f"{class_name}.npy")

    # Check local cache first
    if os.path.exists(cache_path):
        try:
            cached_data = np.load(cache_path)
            if cached_data.shape[0] >= n_samples:
                print(f"  -> Found cached '{class_name}.npy' ({cached_data.shape[0]} drawings)")
                return cached_data[:n_samples]
        except Exception:
            pass

    # Fetch from Google Cloud Storage via HTTP Range request
    url = f"{BASE_URL}/{class_name}.npy"
    # npy header is 80 bytes for QuickDraw format (1, 0)
    header_offset = 80
    end_byte = header_offset + (n_samples * 784) - 1

    print(f"  -> Fetching {n_samples} real sketches for '{class_name}' ({((end_byte+1)/1024/1024):.2f} MB)...", end=" ")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Range": f"bytes=0-{end_byte}"
        }
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_bytes = resp.read()

        # Extract data payload after header
        data_payload = raw_bytes[header_offset : header_offset + (n_samples * 784)]
        data = np.frombuffer(data_payload, dtype=np.uint8).reshape((-1, 784))
        
        # Save to cache
        np.save(cache_path, data)
        print(f"Done in {time.time()-t0:.1f}s ({data.shape[0]} samples)")
        return data
    except Exception as e:
        print(f"\n[WARNING] Range download failed for '{class_name}': {e}. Trying full download fallback...")
        try:
            file_path, _ = urllib.request.urlretrieve(url, cache_path)
            data = np.load(file_path, mmap_mode="r")
            selected = np.array(data[:n_samples], dtype=np.uint8)
            np.save(cache_path, selected)
            return selected
        except Exception as err:
            print(f"[ERROR] Could not download data for '{class_name}': {err}")
            return None


def load_all_classes(classes, samples_per_class=4000):
    """
    Loads real Quick, Draw! data across all 10 target classes.
    """
    x_list = []
    y_list = []

    print(f"\n[DATASET] Loading {samples_per_class} real human drawings per class ({len(classes)} classes)...")
    for idx, class_name in enumerate(classes):
        print(f"[{idx+1}/{len(classes)}] Class: '{class_name}'")
        data = download_class_samples(class_name, n_samples=samples_per_class)
        if data is None:
            raise RuntimeError(f"Failed to acquire data for class: {class_name}")

        labels = np.full((len(data),), idx, dtype=np.int32)
        x_list.append(data)
        y_list.append(labels)

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return x_all, y_all


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Constructs an optimized Convolutional Neural Network (CNN) tailored for sketch contours.
    Uses BatchNormalization and Dropout for strong generalization and high real-world accuracy.
    """
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.20),

        # Second Conv Block
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Dense Classifier Head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.40),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_and_save_history(history, output_path="training_history.png"):
    """Plots and saves loss and accuracy curves."""
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("accuracy", []), label="Train Accuracy", color="#D35400", lw=2)
    plt.plot(history.history.get("val_accuracy", []), label="Val Accuracy", color="#2C2C2C", lw=2, linestyle="--")
    plt.title("DoodleNet - Accuracy Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []), label="Train Loss", color="#D35400", lw=2)
    plt.plot(history.history.get("val_loss", []), label="Val Loss", color="#2C2C2C", lw=2, linestyle="--")
    plt.title("DoodleNet - Loss Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\n[PLOT] Training history saved to '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Train DoodleNet CNN on real Quick, Draw! dataset.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs (default: 8)")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size (default: 128)")
    parser.add_argument("--samples-per-class", type=int, default=4000,
                        help="Number of drawings per class (default: 4000)")
    parser.add_argument("--output-model", type=str, default="model.h5", help="Path to save model (default: model.h5)")
    args = parser.parse_args()

    print("=" * 65)
    print("      DoodleNet: Real-Time Sketch Recognition - Model Training")
    print("=" * 65)
    print(f"Classes ({len(CLASSES)}): {', '.join(CLASSES)}")
    print(f"Samples per class: {args.samples_per_class} (Total: {args.samples_per_class * len(CLASSES)})")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 65)

    # 1. Load Real Dataset
    x_raw, y_raw = load_all_classes(CLASSES, samples_per_class=args.samples_per_class)

    # 2. Preprocess Data: Reshape to (N, 28, 28, 1) and normalize to [0, 1]
    print("\n[PREPROCESSING] Normalizing to [0.0, 1.0] and reshaping to (28, 28, 1)...")
    x = (x_raw / 255.0).reshape(-1, 28, 28, 1).astype(np.float32)
    y = to_categorical(y_raw, num_classes=len(CLASSES))

    # 3. Stratified Shuffle and 85/15 Train-Test split
    np.random.seed(42)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    split_idx = int(0.85 * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  -> Total Samples : {len(x)}")
    print(f"  -> Train Samples : {len(x_train)}")
    print(f"  -> Test Samples  : {len(x_test)}")

    # 4. Build Model
    print("\n[MODEL] Compiling CNN Architecture...")
    model = build_cnn_model(input_shape=(28, 28, 1), num_classes=len(CLASSES))
    model.summary()

    callback_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]

    # 5. Train Model
    print(f"\n[TRAINING] Starting training for {args.epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callback_list,
        verbose=1
    )

    # 6. Evaluate on Test Set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[EVALUATION] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")

    # 7. Save Model and Class Metadata
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_model)
    model.save(model_path)
    print(f"\n[SUCCESS] Trained model saved to '{model_path}'")

    classes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classes.json")
    with open(classes_path, "w") as f:
        json.dump({
            "classes": CLASSES,
            "num_classes": len(CLASSES),
            "input_shape": [28, 28, 1],
            "test_accuracy": round(float(test_acc * 100), 2)
        }, f, indent=2)
    print(f"[SUCCESS] Class metadata saved to '{classes_path}'")

    # 8. Save Loss/Accuracy Curve
    plot_and_save_history(history, output_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_history.png"))

    print("\n" + "=" * 65)
    print("Training finished! You can now test in Streamlit:")
    print("  streamlit run app.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
