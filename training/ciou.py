import tensorflow as tf

def compute_ciou(pred_boxes, true_boxes):
    """
    pred_boxes, true_boxes: Tensor der Form [batch, 4] -> [x_center, y_center, width, height]
    """
    # IoU berechnen
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    true_x1 = true_boxes[:, 0] - true_boxes[:, 2] / 2
    true_y1 = true_boxes[:, 1] - true_boxes[:, 3] / 2
    true_x2 = true_boxes[:, 0] + true_boxes[:, 2] / 2
    true_y2 = true_boxes[:, 1] + true_boxes[:, 3] / 2

    # Intersection
    inter_x1 = tf.maximum(pred_x1, true_x1)
    inter_y1 = tf.maximum(pred_y1, true_y1)
    inter_x2 = tf.minimum(pred_x2, true_x2)
    inter_y2 = tf.minimum(pred_y2, true_y2)
    inter_area = tf.maximum(inter_x2 - inter_x1, 0) * tf.maximum(inter_y2 - inter_y1, 0)

    # Union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    union_area = pred_area + true_area - inter_area

    iou = inter_area / (union_area + 1e-7)

    # Mittelpunkt-Distanz
    center_dist = tf.square(pred_boxes[:, 0] - true_boxes[:, 0]) + tf.square(pred_boxes[:, 1] - true_boxes[:, 1])

    # Enclosing box
    enclose_x1 = tf.minimum(pred_x1, true_x1)
    enclose_y1 = tf.minimum(pred_y1, true_y1)
    enclose_x2 = tf.maximum(pred_x2, true_x2)
    enclose_y2 = tf.maximum(pred_y2, true_y2)
    enclose_diag = tf.square(enclose_x2 - enclose_x1) + tf.square(enclose_y2 - enclose_y1)

    # Aspect ratio
    v = (4 / (3.141592653589793 ** 2)) * tf.square(
        tf.atan(true_boxes[:, 2] / (true_boxes[:, 3] + 1e-7)) -
        tf.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + 1e-7))
    )
    alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - (center_dist / (enclose_diag + 1e-7) + alpha * v)
    ciou_loss = 1 - ciou
    return ciou_loss
