from ultralytics import YOLO
import cv2

# === Modell laden ===
model = YOLO("model_output/best.pt")

# === Klassennamen  ===
class_names = {
    0: "nop",
    1: "index",
    2: "index_pinky",
    3: "pinky",
    4: "thumb",
    5: "thumb_index",
    6: "offen"

}

# === Kamera starten (0 = Standardkamera) ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Konnte Kamera nicht öffnen.")
    exit()

print(" Kamera gestartet. Drücke 'q' zum Beenden.")

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Fehler beim Lesen des Frames.")
        break

    # Inferenz durchführen
    results = model(frame)[0]

    # Ergebnisse durchgehen
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = class_names.get(cls_id, f"Klasse {cls_id}")
        
        # Bounding Box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label + Confidence
        label = f"{name} ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 2)

    # Bild anzeigen
    cv2.imshow("YOLO Handzeichen-Erkennung", frame)

    # Beenden mit 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Aufräumen
cap.release()
cv2.destroyAllWindows()
