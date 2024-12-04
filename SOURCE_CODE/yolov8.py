import numpy as np
import cv2
from ultralytics import YOLO
import random
import pyttsx3  # Library for text-to-speech

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# opening the file in read mode
my_file = open("coco.txt", "r")
# reading the file
data = my_file.read()
# replacing and splitting the text by newline ('\n').
class_list = data.split("\n")
my_file.close()

# Generate random colors for each class for visualization
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Frame dimensions for resizing (optional)
frame_wid = 640
frame_hyt = 480

# Open the video capture (from a file or a camera)
cap = cv2.VideoCapture(0)  # Or 0 for webcam

if not cap.isOpened():
    print("Cannot open camera or video")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Predict on the frame using YOLOv8
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()

    # Check if detections exist in the frame
    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]  # Get a specific box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            # Draw bounding box around the object
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Get the class name from the class list
            detected_class = class_list[int(clsID)]

            # Display class name and confidence on the video frame
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                detected_class + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

            # Text-to-speech: Speak out the detected class
            engine.say(f"Detected {detected_class}")
            engine.runAndWait()

    # Display the resulting frame
    cv2.imshow('Object Detection with YOLOv8', frame)

    # Terminate the program when "Q" is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and destroy OpenCV windows
cap.release()
cv2.destroyAllWindows()
engine.stop()  # Stop the text-to-speech engine
