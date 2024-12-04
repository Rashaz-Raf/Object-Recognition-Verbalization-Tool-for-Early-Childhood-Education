import cv2
import numpy as np
import tensorflow as tf

from keras.applications.resnet50 import preprocess_input, ResNeqqqt50, decode_predictions
from tensorflow.keras.preprocessing import image
import pyttsx3
from PIL import Image
import time

# Load the ResNet50 model pre-trained on ImageNet dataset
model = ResNet50(weights='imagenet')

# Initialize text-to-speech engine
engine = pyttsx3.init()

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))  # ResNet50 expects 224x224 image input
    img = Image.fromarray(img)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def detect_and_speak(frame):
    # Preprocess the frame for ResNet50 model
    preprocessed_frame = preprocess_frame(frame)

    # Predict using ResNet50 model
    predictions = model.predict(preprocessed_frame)

    # Decode the predictions into class labels
    decoded_predictions = decode_predictions(predictions, top=1)[0]  # Get top-1 prediction
    label = decoded_predictions[0][1]  # Get the label name
    confidence = decoded_predictions[0][2]  # Get the confidence score

    # Draw the prediction label and confidence on the frame
    cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Speak out the prediction
    engine.say(f"I see {label}")
    engine.runAndWait()

    return frame


def run_camera():
    cap = cv2.VideoCapture(0)  # Use camera, 0 is the default camera index

    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Detect objects and speak out the labels
        frame_with_detection = detect_and_speak(frame)

        # Display the resulting frame
        cv2.imshow('Real-Time Object Recognition', frame_with_detection)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


# Run the real-time object recognition
run_camera()
