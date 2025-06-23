from ultralytics import YOLO
import cv2

# === Load trained YOLO model ===
model = YOLO("model_output/best.pt")

# === Class ID to name mapping ===
class_names = {
    0: "nop",
    1: "index",
    2: "index_pinky",
    3: "pinky",
    4: "thumb",
    5: "thumb_index",
    6: "piu"
}

# === Start webcam (device 0 = default camera) ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Failed to open camera.")
    exit()

print(" Camera started. Press 'q' to quit.")

# === Main loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Failed to read frame from camera.")
        break

    # Run YOLOv8 inference
    results = model(frame)[0]

    # Process detection results
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = class_names.get(cls_id, f"class {cls_id}")

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label with confidence
        label = f"{name} ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show annotated frame
    cv2.imshow("YOLO Hand Sign Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
