from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt") 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Can't access webcam!")
    exit()

print("The AI ​​is starting... Press 'Q' to exit.")

while True:
    success, frame = cap.read()
    
    if not success:
        print("I can't read the image from the camera..")
        break

    results = model(frame, stream=True)

    for r in results:
        annotated_frame = r.plot() 
        cv2.imshow("YOLO AI Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
