import torch
from ultralytics import YOLO
import cv2
import numpy as np
import requests
import speech_recognition as sr
from gtts import gTTS
import pygame
import os

model = YOLO("yolov8n.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

url = "http://192.168.18.18:8080/shot.jpg"

recognizer = sr.Recognizer()
wake_word = "what is this"

pygame.mixer.init()

def speak(text):
    """Converts text to speech and plays it"""
    tts = gTTS(text=text, lang='en')
    filename = "response.mp3"
    tts.save(filename)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass
    pygame.mixer.music.unload()
    os.remove(filename)

def recognize_objects(img):
    """Detect objects in the provided image frame"""
    results = model(img)
    detections = results[0].boxes.data
    for det in detections:
        label = model.names[int(det[5])]
        return label
    return None

def get_camera_frame():
    """Fetches image from IP camera"""
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    return img

def listen_for_command():
    """Listens for the wake word to trigger object detection"""
    with sr.Microphone() as source:
        print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio).lower()
            if wake_word in command:
                print("Command received:", command)
                return True
            else:
                print("Wake word not detected.")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError:
            print("Error with speech recognition service")
    return False

print("System is ready. Say the wake word to start object detection.")
while True:
    if listen_for_command():
        print("Starting object detection...")

        frame = get_camera_frame()
        detected_object = recognize_objects(frame)

        if detected_object:
            output_text = f"I see {detected_object}"
            print(output_text)
            speak(output_text)
        else:
            print("No object detected.")
            speak("I see nothing")

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
pygame.mixer.quit()








