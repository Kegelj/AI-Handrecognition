import tensorflow as tf
import numpy as np

class HandboxLosses:
    def __init__(self):
        pass  # falls später Parameter gebrauchst werden

    def ciou_loss(self, y_true, y_pred):
        x_true, y_true_c, w_true, h_true = y_true[:, 1], y_true[:, 2], y_true[:, 3], y_true[:, 4]
        x_pred, y_pred_c, w_pred, h_pred = y_pred[:, 1], y_pred[:, 2], y_pred[:, 3], y_pred[:, 4]

        x1_true, y1_true = x_true - w_true / 2, y_true_c - h_true / 2
        x2_true, y2_true = x_true + w_true / 2, y_true_c + h_true / 2

        x1_pred, y1_pred = x_pred - w_pred / 2, y_pred_c - h_pred / 2
        x2_pred, y2_pred = x_pred + w_pred / 2, y_pred_c + h_pred / 2

        inter_x1 = tf.maximum(x1_true, x1_pred)
        inter_y1 = tf.maximum(y1_true, y1_pred)
        inter_x2 = tf.minimum(x2_true, x2_pred)
        inter_y2 = tf.minimum(y2_true, y2_pred)
        inter_area = tf.maximum(0.0, inter_x2 - inter_x1) * tf.maximum(0.0, inter_y2 - inter_y1)

        area_true = w_true * h_true
        area_pred = w_pred * h_pred
        union_area = area_true + area_pred - inter_area + 1e-7
        iou = inter_area / union_area

        center_dist = tf.square(x_pred - x_true) + tf.square(y_pred_c - y_true_c)
        enclose_x1 = tf.minimum(x1_true, x1_pred)
        enclose_y1 = tf.minimum(y1_true, y1_pred)
        enclose_x2 = tf.maximum(x2_true, x2_pred)
        enclose_y2 = tf.maximum(y2_true, y2_pred)
        c2 = tf.square(enclose_x2 - enclose_x1) + tf.square(enclose_y2 - enclose_y1) + 1e-7

        v = (4 / (np.pi ** 2)) * tf.square(tf.atan(w_true / (h_true + 1e-7)) - tf.atan(w_pred / (h_pred + 1e-7)))
        alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - center_dist / c2 - alpha * v
        loss = 1 - ciou
        mask = tf.cast(tf.equal(y_true[:, 0], 1.0), tf.float32)
        return tf.reduce_mean(mask * loss)

    def final_loss(self, y_true, y_pred):
        class_loss = tf.keras.losses.binary_crossentropy(y_true[:, 0], y_pred[:, 0])
        bbox_ciou_loss = self.ciou_loss(y_true, y_pred)
        return class_loss + bbox_ciou_loss

    def box_metrics(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        center_dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        area1 = w1 * h1
        area2 = w2 * h2
        size_ratio = area1 / area2 if area2 > 0 else 0
        size_error = abs(1 - size_ratio)

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

        return center_dist, size_error, overlap
