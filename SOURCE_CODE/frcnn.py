import torch
import cv2
import pyttsx3
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

engine = pyttsx3.init()

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame_tensor = F.to_tensor(frame)

        with torch.no_grad():
            predictions = model([frame_tensor])[0]

        boxes = predictions['boxes']
        labels = predictions['labels']
        scores = predictions['scores']

        threshold = 0.5
        for i, score in enumerate(scores):
            if score > threshold:
                label = COCO_INSTANCE_CATEGORY_NAMES[labels[i].item()]

                box = boxes[i].cpu().numpy().astype(int)

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                engine.say(f"I see a {label}")
                engine.runAndWait()

        cv2.imshow('Faster R-CNN', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
