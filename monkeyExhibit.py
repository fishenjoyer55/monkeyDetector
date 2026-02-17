import torch
import os
import sys
import cv2, time
import numpy as np
from PIL import Image

from monkeyLab import MonkeyNet
from monkeyLab import transform
from monkeyLab import dataloader
from monkeyLab import dataset

device = "mps" if torch.backends.mps.is_available() else "cpu"
net = MonkeyNet().to(device)
if os.path.exists("model_weights.pth"):
    print("Existing monkey model found. Commencing exhibit.")
    net.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
else:
    print("We are highkey monkeyless. Go create and train a monkey model first.")
    sys.exit()
correct = 0
total = 0

with torch.no_grad():
    for data in dataloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy: %f" % (correct/total))
if input("Begin monkey exhibit? [y/n] ") != "y":
    sys.exit()

capture = cv2.VideoCapture(0)
capture.set(3, 480)
capture.set(4, 480)
print("Beginning monkey exhibit.")
cv2.namedWindow("Monkey exhibit", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Monkey exhibit", 960, 480)

expressions = dataset.classes
monkey_faces = {}
for expression in expressions:
    img_path = f"monkeys/{expression}.jpg"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            monkey_faces[expression] = img
        else:
            print(f"{img_path} could not be decoded")
    else:
        print(f"Missing {img_path}")


while True:
    ret, frame = capture.read()
    if not ret:
        break

    image_tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)

    with torch.no_grad():
        output = net(image_tensor)
        probabilities = torch.softmax(output, dim = 1)
        confidence, prediction = torch.max(probabilities, dim = 1)

    confidence_value = confidence.item()
    decision = expressions[prediction.item()] if confidence > 0.6 else "neutral"
    display_image = cv2.resize(monkey_faces.get(decision), (480, 480))

    if display_image is not None:
        combined = np.hstack((frame, display_image))
    else:
        combined = frame

    text = f"{decision} ({confidence_value*100:.1f}%)"
    cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Monkey exhibit", combined)

    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()