import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#  Parameter
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 200
max_shift_ratio = 0.2

#  Eigene OpenCV-Augmentierungen
def random_brightness_contrast(image, brightness_range=(-30, 30), contrast_range=(0.8, 1.2)):
    alpha = np.random.uniform(contrast_range[0], contrast_range[1])
    beta = np.random.uniform(brightness_range[0], brightness_range[1])
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def random_hue_saturation(image, hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int32)
    hsv_image[..., 0] += np.random.randint(-hue_shift_limit, hue_shift_limit)
    hsv_image[..., 1] += np.random.randint(-sat_shift_limit, sat_shift_limit)
    hsv_image[..., 2] += np.random.randint(-val_shift_limit, val_shift_limit)
    hsv_image = np.clip(hsv_image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def random_missing_pixels(image, amount=0.1):
    noisy_image = image.copy()
    num_pixels = int(amount * image.size / 3)
    coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1], :] = 0
    return noisy_image

def random_color_from_palette():
    palette = [
        [255, 255, 0], [255, 0, 0], [0, 0, 255],
        [0, 255, 0], [255, 165, 0], [128, 0, 128]
    ]
    return palette[np.random.randint(len(palette))]

def randomize_dark_and_bright_pixels_palette(image, dark_threshold=30, bright_threshold=220):
    image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dark_mask = gray < dark_threshold
    bright_mask = gray > bright_threshold
    combined_mask = np.logical_or(dark_mask, bright_mask)
    coords = np.column_stack(np.where(combined_mask))
    for y, x in coords:
        image[y, x] = random_color_from_palette()
    return image

def random_blur(image, max_ksize=5):
    if np.random.rand() < 0.3:
        ksize = np.random.choice([3, 5])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    return image

def random_rotation(image, bbox, max_angle=10):
    if np.random.rand() >= 0.3:
        return image, bbox
    h, w, _ = image.shape
    angle = np.random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated, bbox

def random_cutout(image, max_fraction=0.2):
    if np.random.rand() >= 0.3:
        return image
    h, w, _ = image.shape
    cutout_w = int(w * max_fraction)
    cutout_h = int(h * max_fraction)
    x1 = np.random.randint(0, w - cutout_w)
    y1 = np.random.randint(0, h - cutout_h)
    image[y1:y1 + cutout_h, x1:x1 + cutout_w, :] = 0
    return image

def random_wrap_shift_no_bbox_clipping(image, bbox):
    h, w, _ = image.shape

    # Tensor -> NumPy
    bbox = bbox.numpy() if isinstance(bbox, tf.Tensor) else bbox

    # Bounding Box in Pixel
    x_c, y_c, bw, bh = bbox[1] * w, bbox[2] * h, bbox[3] * w, bbox[4] * h
    x_min = x_c - bw / 2
    x_max = x_c + bw / 2
    y_min = y_c - bh / 2
    y_max = y_c + bh / 2

    # Erlaubter Shift
    max_dx = min(int(w * max_shift_ratio), int(x_min))
    min_dx = -min(int(w * max_shift_ratio), int(w - x_max))
    max_dy = min(int(h * max_shift_ratio), int(y_min))
    min_dy = -min(int(h * max_shift_ratio), int(h - y_max))

    dx = np.random.randint(min_dx, max_dx + 1)
    dy = np.random.randint(min_dy, max_dy + 1)

    # Wrap-around Verschiebung
    shifted_image = np.roll(image, shift=dx, axis=1)
    shifted_image = np.roll(shifted_image, shift=dy, axis=0)

    # Neue BBox-Koordinaten berechnen (nur wenn Hand vorhanden)
    if bbox[0] == 1:
        bbox = bbox.copy()  # NumPy-Array kopieren
        bbox[1] = (x_c + dx) / w
        bbox[2] = (y_c + dy) / h

    return shifted_image, bbox


def opencv_augment(img, bbox):
    img = (img.numpy() * 255).astype(np.uint8)

    if np.random.rand() < 0.3:
        img, bbox = random_wrap_shift_no_bbox_clipping(img, bbox)
    if np.random.rand() < 0.5:
        img = random_brightness_contrast(img)
    if np.random.rand() < 0.5:
        img = random_hue_saturation(img)
    if np.random.rand() < 0.1:
        img = random_missing_pixels(img)
    if np.random.rand() < 0.1:
        img = randomize_dark_and_bright_pixels_palette(img)
    img = random_blur(img)
    img, bbox = random_rotation(img, bbox)
    img = random_cutout(img)
    img = img.astype(np.float32) / 255.0
    return img, bbox

def tf_opencv_augment(img, bbox):
    img, bbox = tf.py_function(func=opencv_augment, inp=[img, bbox], Tout=(tf.float32, tf.float32))
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    bbox.set_shape([5])
    return img, bbox

#  Labels laden
def load_image_and_label(image_path):
    txt_path = image_path.with_suffix(".txt")
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    with open(txt_path) as f:
        cls, x_c, y_c, w, h = map(float, f.readline().split())
        if cls == 0:
            label = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        else:
            label = np.array([1, x_c, y_c, w, h], dtype=np.float32)
    return img.astype(np.float32), label

#  Dataset-Erstellung
def create_tf_dataset(image_paths, augment_data=False):
    def generator():
        for path in image_paths:
            yield load_image_and_label(path)
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(5,), dtype=tf.float32)
        )
    )
    if augment_data:
        dataset = dataset.map(tf_opencv_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(1)

#  Bildpfade laden und Split
all_data_dir = Path("training/alle_daten")
all_imgs = sorted([p for p in all_data_dir.rglob("*.jpg") if (p.with_suffix(".txt")).exists()])
train_imgs, test_imgs = train_test_split(all_imgs, test_size=0.2, random_state=42, shuffle=True)
print(f" Train: {len(train_imgs)} Bilder, Test: {len(test_imgs)} Bilder")

#  Daten laden
train_ds = create_tf_dataset(train_imgs, augment_data=True)
test_ds = create_tf_dataset(test_imgs, augment_data=False)

#  Modell mit 5 Outputs (class_prob + bbox)
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
    x = tf.keras.layers.Dense(64, activation='relu')(x)  # Zwischenschritt
    outputs = tf.keras.layers.Dense(5, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)


def iou_metric(y_true, y_pred):
    x1_true = y_true[:, 1] - y_true[:, 3] / 2
    y1_true = y_true[:, 2] - y_true[:, 4] / 2
    x2_true = y_true[:, 1] + y_true[:, 3] / 2
    y2_true = y_true[:, 2] + y_true[:, 4] / 2

    x1_pred = y_pred[:, 1] - y_pred[:, 3] / 2
    y1_pred = y_pred[:, 2] - y_pred[:, 4] / 2
    x2_pred = y_pred[:, 1] + y_pred[:, 3] / 2
    y2_pred = y_pred[:, 2] + y_pred[:, 4] / 2

    inter_x1 = tf.maximum(x1_true, x1_pred)
    inter_y1 = tf.maximum(y1_true, y1_pred)
    inter_x2 = tf.minimum(x2_true, x2_pred)
    inter_y2 = tf.minimum(y2_true, y2_pred)

    inter_area = tf.maximum(0.0, inter_x2 - inter_x1) * tf.maximum(0.0, inter_y2 - inter_y1)
    true_area = (x2_true - x1_true) * (y2_true - y1_true)
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)

    union_area = true_area + pred_area - inter_area + 1e-7
    iou = inter_area / union_area

    return tf.reduce_mean(1 - iou)
def box_metrics(box1, box2):
    # box = (x_center, y_center, width, height)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Mittelpunkt-Distanz
    center_dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Flächen
    area1 = w1 * h1
    area2 = w2 * h2
    size_ratio = area1 / area2 if area2 > 0 else 0

    # Überlappung prüfen (Intersection Area > 0)
    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2

    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    overlap = inter_area > 0

    return center_dist, size_ratio, overlap

#  Neue Loss-Funktion (kombiniert mehrere Aspekte)
def custom_loss(y_true, y_pred):
    class_loss = tf.keras.losses.binary_crossentropy(y_true[:, 0], y_pred[:, 0])
    mask = tf.cast(tf.equal(y_true[:, 0], 1.0), tf.float32)

    # Center-Distance-Term
    center_loss = tf.sqrt(
        tf.square(y_true[:, 1] - y_pred[:, 1]) +
        tf.square(y_true[:, 2] - y_pred[:, 2])
    )
    center_loss = tf.reduce_mean(mask * center_loss)

    # Size-Ratio-Term
    true_area = y_true[:, 3] * y_true[:, 4]
    pred_area = y_pred[:, 3] * y_pred[:, 4] + 1e-7
    size_ratio = true_area / pred_area
    size_loss = tf.reduce_mean(mask * tf.abs(1 - size_ratio))

    # IoU-Term
    iou_loss = iou_metric(y_true, y_pred)

    # Kombinierte Loss: Gewichte kannst du anpassen!
    total_loss = class_loss + 0.5 * center_loss + 0.3 * size_loss + 0.2 * iou_loss

    return total_loss

#  Modell & Training
model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=custom_loss)

mse_history, acc_history = [], []

best_center_match = 1e10  # kleinster Center-Distance
Path("model_output").mkdir(parents=True, exist_ok=True)



best_score = 1e10  # Startwert

for epoch in range(EPOCHS):
    print(f"\n📦 Epoch {epoch+1}/{EPOCHS}")
    model.fit(train_ds, epochs=1, verbose=1, validation_data=test_ds)

    y_true, y_pred = [], []
    for imgs, labels in test_ds:
        preds = model.predict(imgs)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    mse = mean_squared_error(y_true, y_pred)
    hits, total = 0, 0
    center_dists, size_ratios, overlaps = [], [], []
    for p, t in zip(y_pred, y_true):
        if t[0] == 1:  # nur wenn Hand vorhanden
            total += 1
            center_dist, size_ratio, overlap = box_metrics(p[1:], t[1:])
            center_dists.append(center_dist)
            size_ratios.append(size_ratio)
            overlaps.append(overlap)
            if center_dist < 0.05 and 0.8 <= size_ratio <= 1.2 and overlap:
                hits += 1

    acc = hits / total if total > 0 else 0.0

    mse_history.append(mse)
    acc_history.append(acc)
    mean_center = np.mean(center_dists) if center_dists else 0
    mean_size = np.mean(size_ratios) if size_ratios else 0
    mean_overlap = np.mean(overlaps) if overlaps else 0

    # **Neu: Kombinierter Score** – je kleiner, desto besser!
    mean_size_error = abs(1 - mean_size)
    combined_score = (0.5 * mean_center) + (0.3 * mean_size_error) + (0.2 * (1 - mean_overlap))

    print(f" Test MSE: {mse:.5f}")
    print(f" Center Distance ⌀: {mean_center:.4f}")
    print(f" Size Ratio ⌀: {mean_size:.2f}")
    print(f" Overlap-Rate: {mean_overlap:.2%}")
    print(f" Genauigkeit (alle Bedingungen erfüllt): {acc:.2%}")
    print(f" Kombinierter Score: {combined_score:.5f}")

    # Speicher immer das Modell mit Epochennummer
    model.save(f"model_output/handbox_model_epoch_{epoch+1}.h5")
    print(f" Modell für Epoche {epoch+1} gespeichert.")
    
    # Speicher das Modell, wenn der kombinierte Score besser ist
    if combined_score < best_score:
        best_score = combined_score
        model.save("model_output/best_handbox_model.h5")
        print(f" Neues bestes Modell gespeichert (kombinierter Score: {combined_score:.5f})")
