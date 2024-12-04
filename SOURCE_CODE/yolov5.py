import cv2
print(cv2.__version__)

import cv2
import torch
import pyttsx3

engine = pyttsx3.init()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        results = model(frame)

        labels = results.xyxyn[0][:, -1].numpy()
        classes = results.names
        detected_objects = [classes[int(label)] for label in labels]

        cv2.imshow('YOLOv5', frame)

        if detected_objects:
            detected_text = f"I see {', '.join(detected_objects)}."
            engine.say(detected_text)
            engine.runAndWait()
            print(detected_text)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
