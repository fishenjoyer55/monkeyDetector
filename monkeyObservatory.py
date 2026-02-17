import cv2
import os
import time
import numpy as np

expressions = ["confused", "eureka", "neutral"]
for expression in expressions:
    os.makedirs(os.path.join("images", expression), exist_ok = True)

capture = cv2.VideoCapture(0)
capture.set(3, 480)
capture.set(4, 480)

print("Starting monkey observation.\n")
print("Copy the pose of the monkey shown on screen. For each expression, a countdown will begin, and then 50 frames will be captured.")


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

cv2.namedWindow("Monkey Observer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Monkey Observer", 1280, 480)

for expression in expressions:
    print(f"Prepare for: {expression}")

    count = 0
    while count < 70:
        time.sleep(0.1)
        ret, frame = capture.read()
        if not ret:
            break

        display_image = cv2.resize(monkey_faces.get(expression), (480, 480))
        combined = np.hstack((frame, display_image))

        if count > 20:
            filepath = os.path.join("images", expression, f"{expression}_{count}.jpg")
            cv2.imwrite(filepath, frame)
        if count == 20:
            print("Recording.")

        cv2.imshow("Monkey Observer", combined)

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

        count += 1

    print(f"Captured {count} images for {expression}\n")
    time.sleep(1)

print("Monkey observed. Proceed to monkey business.")
capture.release()
cv2.destroyAllWindows()
