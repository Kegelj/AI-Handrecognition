import tensorflow as tf
import numpy as np
import cv2

class MyImageAugmentor:
    def __init__(self, img_size=(256, 256), max_shift_ratio=0.1):
        self.img_size = img_size
        self.max_shift_ratio = max_shift_ratio

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
            img_np = img_np.numpy()  # Tensor → NumPy-Array
            return cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0).astype(np.float32)
        img = tf.py_function(gaussian_blur, [img], tf.float32)
        img.set_shape([self.img_size[0], self.img_size[1], 3])
        return img

    def tf_random_cutout(self, img, bbox, max_fraction=0.2):
        def cutout_np(img_np, bbox_np):
            img_np = img_np.numpy()  # Tensor → NumPy-Array
            h, w, _ = img_np.shape
            cutout_w = int(w * max_fraction)
            cutout_h = int(h * max_fraction)

        # Bounding Box in Pixelkoordinaten
            x_c, y_c, bw, bh = bbox_np[1] * w, bbox_np[2] * h, bbox_np[3] * w, bbox_np[4] * h
            x_min = int(x_c - bw / 2)
            x_max = int(x_c + bw / 2)
            y_min = int(y_c - bh / 2)
            y_max = int(y_c + bh / 2)

            tries = 0
            while True:
            # Zufällige linke obere Ecke
                x1 = np.random.randint(0, w - cutout_w)
                y1 = np.random.randint(0, h - cutout_h)
                x2 = x1 + cutout_w
                y2 = y1 + cutout_h

            # Überprüfen, ob sich der Cutout-Bereich mit der BBox überschneidet
                if x2 <= x_min or x1 >= x_max or y2 <= y_min or y1 >= y_max:
                # Kein Überlapp – perfekt!
                    img_np[y1:y2, x1:x2, :] = 0
                    break

                tries += 1
                if tries > 100:
                # Fallback: nach 100 Versuchen – trotzdem platzieren (damit Code nie hängt!)
                    img_np[y1:y2, x1:x2, :] = 0
                    break

            return img_np.astype(np.float32)

        img = tf.py_function(cutout_np, [img, bbox], tf.float32)
        img.set_shape([self.img_size[0], self.img_size[1], 3])
        return img


    def randomize_dark_and_bright_pixels_palette(self, image, dark_threshold=30, bright_threshold=220, palette=None):
        image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dark_mask = gray < dark_threshold
        bright_mask = gray > bright_threshold
        combined_mask = np.logical_or(dark_mask, bright_mask)
        coords = np.column_stack(np.where(combined_mask))
        if palette is None:
            palette = [
                [255, 255, 0], [255, 0, 0], [0, 0, 255],
                [0, 255, 0], [255, 165, 0], [128, 0, 128]
            ]
        for y, x in coords:
            image[y, x] = palette[np.random.randint(len(palette))]
        return image

    # Kombinierte Augmentation
    def tf_augment(self, img, bbox):
        img = self.tf_random_brightness_contrast(img)
        img = self.tf_random_hue_saturation(img)
        img = self.tf_random_blur(img)
        img = self.tf_random_cutout(img,bbox)

        # 🚫 Teilweise deaktiviert: random_wrap_shift_no_bbox_clipping
        # def wrap_shift_opencv(img_np, bbox_np):
        #     img_shifted, bbox_shifted = self.random_wrap_shift_no_bbox_clipping(img_np, bbox_np)
        #     img_shifted = img_shifted.astype(np.float32) / 255.0
        #     return img_shifted, bbox_shifted
        #
        # img, bbox = tf.py_function(wrap_shift_opencv, [img * 255.0, bbox], [tf.float32, tf.float32])
        # img.set_shape([self.img_size[0], self.img_size[1], 3])
        # bbox.set_shape([5])
        return img, bbox









    # OpenCV-Only: Random Wrap Shift
    # def random_wrap_shift_no_bbox_clipping(self, image, bbox):
    #     h, w, _ = image.shape
    #     bbox = bbox.numpy() if isinstance(bbox, tf.Tensor) else bbox

    #     x_c, y_c, bw, bh = bbox[1] * w, bbox[2] * h, bbox[3] * w, bbox[4] * h
    #     x_min = x_c - bw / 2
    #     x_max = x_c + bw / 2
    #     y_min = y_c - bh / 2
    #     y_max = y_c + bh / 2

    #     max_dx = min(int(w * self.max_shift_ratio), int(x_min))
    #     min_dx = -min(int(w * self.max_shift_ratio), int(w - x_max))
    #     max_dy = min(int(h * self.max_shift_ratio), int(y_min))
    #     min_dy = -min(int(h * self.max_shift_ratio), int(h - y_max))

    #     dx = np.random.randint(min_dx, max_dx + 1)
    #     dy = np.random.randint(min_dy, max_dy + 1)

    #     shifted_image = np.roll(image, shift=dx, axis=1)
    #     shifted_image = np.roll(shifted_image, shift=dy, axis=0)

    #     if bbox[0] == 1:
    #         bbox = bbox.copy()
    #         bbox[1] = (x_c + dx) / w
    #         bbox[2] = (y_c + dy) / h

    #     return shifted_image, bbox