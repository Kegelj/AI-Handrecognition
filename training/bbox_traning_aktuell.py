import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scr.augments import MyImageAugmentor
from scr.losses import HandboxLosses
from datetime import datetime
from uuid import uuid4
import csv

#  Parameter
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 30
max_shift_ratio = 0.2



unique_id = uuid4()


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
        augmentor = MyImageAugmentor(img_size=IMG_SIZE)
        dataset = dataset.map(augmentor.tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
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
    inputs = tf.keras.Input(shape=(128, 128, 3))

    # Block 1
    x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 2
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 3
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 4
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Dense Layers
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(5, activation='sigmoid')(x)

    return tf.keras.Model(inputs, outputs)




#  Modell & Training
model = build_model()
losses = HandboxLosses()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss=losses.final_loss, metrics=['accuracy'])
mse_history, acc_history = [], []

best_center_match = 1e10  # kleinster Center-Distance
Path("model_output").mkdir(parents=True, exist_ok=True)



best_score = 1e10  # Startwert

csv_file = f"model_output/{unique_id}.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "training",
            "epoch",
            "current_time_start",
            "current_time_end",
            "mean_squared_error",
            "mean_center_dist",
            "mean_size_error",
            "mean_overlap",
            "combined_score",
            "acc_all_conditions",
            "train_loss",
            "val_loss",
            "train_acc",
            "val_acc",
            "current_lr",
            "IMG_SIZE",
            "BATCH_SIZE"
        ])


for epoch in range(EPOCHS):
    print(f"\n Epoch {epoch+1}/{EPOCHS}")
    # Zeitmessung Start

    current_time_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history = model.fit(train_ds, epochs=1, verbose=1, validation_data=test_ds)
    current_time_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Zeitmessung Ende

    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    train_acc = history.history.get('accuracy', [None])[-1]
    val_acc = history.history.get('val_accuracy', [None])[-1]
    current_lr = model.optimizer.lr.numpy()



    y_true, y_pred = [], []
    for imgs, labels in test_ds:
        preds = model.predict(imgs)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    mse = mean_squared_error(y_true, y_pred)
    hits, total = 0, 0

    center_dists, size_errors, overlaps = [], [], []
    for p, t in zip(y_pred, y_true):
        if t[0] == 1:
            total += 1
            center_dist, size_error, overlap = losses.box_metrics(p[1:], t[1:])
            center_dists.append(center_dist)
            size_errors.append(size_error)
            overlaps.append(overlap)
            if center_dist < 0.05 and size_error < 0.2 and overlap:
                hits += 1

    acc = hits / total if total > 0 else 0.0
    mean_center = np.mean(center_dists) if center_dists else 0
    mean_size_error = np.mean(size_errors) if size_errors else 0
    mean_overlap = np.mean(overlaps) if overlaps else 0

    combined_score =  mean_center + mean_size_error + mean_overlap

    print(f" Center Distance ⌀: {mean_center:.4f}")
    print(f" Size Error ⌀: {mean_size_error:.2f}")
    print(f" Overlap-Rate: {mean_overlap:.2%}")
    print(f" Genauigkeit (alle Bedingungen erfüllt): {acc:.2%}")
    print(f" Kombinierter Score: {combined_score:.5f}")
    
    # 📌 CSV-Logging in jeder Epoche!
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "training",
            epoch + 1,
            current_time_start,
            current_time_end,
            mse,
            mean_center,
            mean_size_error,
            mean_overlap,
            combined_score,
            acc,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            current_lr,
            str(IMG_SIZE),
            BATCH_SIZE
        ])


    # Modell speichern
    model.save(f"model_output/handbox_model_epoch_{epoch+1}.h5")
    print(f" Modell für Epoche {epoch+1} gespeichert.")
    
    if combined_score < best_score:
        best_score = combined_score
        model.save("model_output/best_handbox_model.h5")
        print(f" Neues bestes Modell gespeichert (kombinierter Score: {combined_score:.5f})")

