import tensorflow as tf
import numpy as np
import cv2

class MyImageAugmentor:
    def __init__(self, img_size=(256, 256), max_shift_ratio=0.1):
        self.img_size = img_size
        self.max_shift_ratio = max_shift_ratio

    def tf_augment(self, img, bbox):
        def _replace_bbox_after_hardcore(img_np, bbox_np):
            img_np = img_np.numpy().astype(np.uint8)
            h, w, _ = img_np.shape

            # BBox-Koordinaten in Pixel
            x_c, y_c = bbox_np[1] * w, bbox_np[2] * h
            bw, bh = bbox_np[3] * w, bbox_np[4] * h
            x1 = int(x_c - bw / 2)
            y1 = int(y_c - bh / 2)
            x2 = int(x_c + bw / 2)
            y2 = int(y_c + bh / 2)

            # --- Sanftes Augmentieren des Bildausschnitts
            img_soft = img_np.copy().astype(np.float32)
            img_soft = cv2.convertScaleAbs(img_soft, alpha=np.random.uniform(0.9, 1.1),
                                           beta=np.random.uniform(-10, 10))
            img_soft = cv2.GaussianBlur(img_soft, (3, 3), 0)

            # BBox-Ausschnitt sichern
            bbox_crop = img_soft[y1:y2, x1:x2, :].copy()

            # --- Hardcore Augment des gesamten Bilds
            img_hard = img_np.copy().astype(np.float32)
            img_hard = cv2.GaussianBlur(img_hard, (21, 21), 5)
            noise = np.random.uniform(0, 70, img_hard.shape).astype(np.float32)
            img_hard = cv2.add(img_hard, noise)
            img_hard = cv2.cvtColor(img_hard.astype(np.uint8), cv2.COLOR_RGB2HSV)
            img_hard[..., 0] = (img_hard[..., 0] + np.random.randint(40)) % 180
            img_hard = cv2.cvtColor(img_hard, cv2.COLOR_HSV2RGB).astype(np.float32)

            # --- Zurückkopieren des BBox-Inhalts
            img_hard[y1:y2, x1:x2, :] = bbox_crop

            return img_hard.astype(np.float32)

        img = tf.py_function(_replace_bbox_after_hardcore, [img, bbox], tf.float32)
        img.set_shape([self.img_size[0], self.img_size[1], 3])

        # Optional: zusätzlicher Cutout außerhalb der BBox
        img = self.tf_random_cutout(img, bbox)

        return img, bbox

    def tf_random_cutout(self, img, bbox, max_fraction=0.2):
        def cutout_np(img_np, bbox_np):
            img_np = img_np.numpy()
            h, w, _ = img_np.shape
            cutout_w = int(w * max_fraction)
            cutout_h = int(h * max_fraction)

            # BBox in Pixel
            x_c = bbox_np[1] * w
            y_c = bbox_np[2] * h
            bw = bbox_np[3] * w
            bh = bbox_np[4] * h
            x_min = int(x_c - bw / 2)
            x_max = int(x_c + bw / 2)
            y_min = int(y_c - bh / 2)
            y_max = int(y_c + bh / 2)

            tries = 0
            while True:
                x1 = np.random.randint(0, w - cutout_w)
                y1 = np.random.randint(0, h - cutout_h)
                x2 = x1 + cutout_w
                y2 = y1 + cutout_h

                # Stelle nur dann cutout, wenn es nicht überlappt
                if x2 <= x_min or x1 >= x_max or y2 <= y_min or y1 >= y_max:
                    img_np[y1:y2, x1:x2, :] = 0
                    break

                tries += 1
                if tries > 100:
                    img_np[y1:y2, x1:x2, :] = 0
                    break

            return img_np.astype(np.float32)

        img = tf.py_function(cutout_np, [img, bbox], tf.float32)
        img.set_shape([self.img_size[0], self.img_size[1], 3])
        return img

    # Optional weitere Augmentierungsmethoden, falls du sie brauchst:
    def tf_random_brightness_contrast(self, img, brightness_delta=0.1, contrast_lower=0.8, contrast_upper=1.2):
        img = tf.image.random_brightness(img, max_delta=brightness_delta)
        img = tf.image.random_contrast(img, lower=contrast_lower, upper=contrast_upper)
        return img

    def tf_random_hue_saturation(self, img, hue_delta=0.05, sat_lower=0.8, sat_upper=1.2):
        img = tf.image.random_hue(img, max_delta=hue_delta)
        img = tf.image.random_saturation(img, lower=sat_lower, upper=sat_upper)
        return img

    def tf_random_blur(self, img, kernel_size=3):
        def gaussian_blur(img_np):
            img_np = img_np.numpy()
            return cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0).astype(np.float32)
        img = tf.py_function(gaussian_blur, [img], tf.float32)
        img.set_shape([self.img_size[0], self.img_size[1], 3])
        return img
