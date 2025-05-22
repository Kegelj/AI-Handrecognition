import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 50

def load_image_and_label(image_path):
    txt_path = image_path.with_suffix(".txt")
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    with open(txt_path) as f:
        cls, x_c, y_c, w, h = map(float, f.readline().split())
        bbox = np.array([x_c, y_c, w, h], dtype=np.float32)
    return img.astype(np.float32), bbox

def augment(img, bbox):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_hue(img, max_delta=0.05)

    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, bbox

def create_tf_dataset(image_paths, augment_data=False):
    def generator():
        for path in image_paths:
            yield load_image_and_label(path)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(4,), dtype=tf.float32)
        )
    )
    if augment_data:
        dataset = dataset.map(augment)
    return dataset.batch(BATCH_SIZE).prefetch(1)

def build_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(4, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)


def calculate_iou(box1, box2):
    x1_min = box1[0] - box1[2]/2
    y1_min = box1[1] - box1[3]/2
    x1_max = box1[0] + box1[2]/2
    y1_max = box1[1] + box1[3]/2
    x2_min = box2[0] - box2[2]/2
    y2_min = box2[1] - box2[3]/2
    x2_max = box2[0] + box2[2]/2
    y2_max = box2[1] + box2[3]/2
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

# Lade Bildpfade
train_dir = Path("training/data_split/train")
test_dir = Path("training/data_split/test")
train_imgs = sorted([p for p in train_dir.glob("*.png") if (p.with_suffix(".txt")).exists()])
test_imgs = sorted([p for p in test_dir.glob("*.png") if (p.with_suffix(".txt")).exists()])

# Datasets
train_ds = create_tf_dataset(train_imgs, augment_data=True)
test_ds = create_tf_dataset(test_imgs, augment_data=False)

# Modell vorbereiten
model = build_model()
model.compile(optimizer='adam', loss='mse')

# Metriken speichern
mse_history = []
acc_history = []

# Training mit Evaluation pro Epoche
for epoch in range(EPOCHS):
    print(f"\n📦 Epoch {epoch+1}/{EPOCHS}")
    model.fit(train_ds, epochs=1, verbose=1)

    y_true, y_pred = [], []
    for imgs, labels in test_ds:
        preds = model.predict(imgs)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    mse = mean_squared_error(y_true, y_pred)
    hits = sum(calculate_iou(p, t) > 0.5 for p, t in zip(y_pred, y_true))
    acc = hits / len(y_true)

    mse_history.append(mse)
    acc_history.append(acc)

    print(f"📉 Test MSE: {mse:.5f}")
    print(f"✅ IoU > 0.5 (BBox Accuracy): {acc:.2%}")

# Finale Bewertung
print("\n🧪 Finale Auswertung auf dem Testset:")
mse = mean_squared_error(y_true, y_pred)
acc = sum(calculate_iou(p, t) > 0.5 for p, t in zip(y_pred, y_true)) / len(y_true)
print(f"✅ Finale MSE: {mse:.5f}")
print(f"✅ Finale IoU > 0.5 (BBox Accuracy): {acc:.2%}")

# Modell speichern
Path("model_output").mkdir(parents=True, exist_ok=True)
model.save("model_output/handbox_model.h5")
print("\n💾 Modell gespeichert unter: model_output/handbox_model.h5")

# Plot speichern
epochs_range = list(range(1, EPOCHS + 1))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, mse_history, marker='o')
plt.title("Test MSE pro Epoche")
plt.xlabel("Epoche")
plt.ylabel("MSE")
plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc_history, marker='o', color='green')
plt.title("BBox Accuracy (IoU > 0.5) pro Epoche")
plt.xlabel("Epoche")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("model_output/training_metrics.png")
plt.show()
