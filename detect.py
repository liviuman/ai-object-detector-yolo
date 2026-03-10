from ultralytics import YOLO
import cv2

# Încarcă modelul
model = YOLO("yolov8n.pt") 

# Deschidem camera. Dacă 0 nu merge, încearcă 1 sau 2.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Eroare: Nu pot accesa camera web!")
    exit()

print("AI-ul pornește... Apasă tasta 'Q' pentru a ieși.")

while True:
    success, frame = cap.read()
    
    if not success:
        print("Nu pot citi imaginea de la cameră.")
        break

    # Rulăm AI-ul pe imagine
    # 'persist=True' ajută la stabilitate
    results = model(frame, stream=True)

    for r in results:
        annotated_frame = r.plot() # Desenează pătrățelele
        cv2.imshow("YOLO AI Detection", annotated_frame) # Arată fereastra

    # Așteaptă 1 milisecundă și verifică dacă ai apăsat 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()