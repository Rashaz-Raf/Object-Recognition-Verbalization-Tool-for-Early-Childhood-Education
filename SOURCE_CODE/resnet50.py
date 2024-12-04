import torch
import torchvision.transforms as T
from torchvision import models
import cv2
import pyttsx3
import time
from PIL import Image

engine = pyttsx3.init()

model = models.resnet50(pretrained=True)
model.eval()

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
import urllib.request, json
with urllib.request.urlopen(LABELS_URL) as url:
    imagenet_labels = json.loads(url.read().decode())

cap = cv2.VideoCapture(0)

fps = 0
frame_count = 0
start_time = time.time()

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame_start_time = time.time()

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)

        _, predicted_idx = torch.max(output, 1)
        predicted_label = imagenet_labels[predicted_idx.item()]

        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        engine.say(f"I see a {predicted_label}")
        engine.runAndWait()

        cv2.imshow('CNN Object Recognition', frame)

        frame_latency = time.time() - frame_start_time
        print(f"Latency: {frame_latency:.4f} seconds")

        frame_count += 1
        current_time = time.time() - start_time
        fps = frame_count / current_time
        print(f"FPS: {fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

end_time = time.time()
total_time = end_time - start_time
print(f"Total Time: {total_time:.2f} seconds")
print(f"Final FPS: {frame_count / total_time:.2f}")
