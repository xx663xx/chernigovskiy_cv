import cv2
import numpy as np
import random
import json
from pathlib import Path
import time

print("Формат ввода: 3 или 4")
n = int(input())

colors = ["red", "green", "blue", "yellow"]
data = {}
clicked = False
pos = [0, 0]

save_path = Path(__file__).parent
config_path = save_path / "config.json"



def mouse(event, x, y, flags, param):
    global clicked, pos
    if event == cv2.EVENT_LBUTTONDOWN:
        pos = [x, y]
        clicked = True

def calibrate(cam):
    global clicked
    cv2.namedWindow("camera")
    cv2.setMouseCallback("camera", mouse)

    for color in colors:
        clicked = False
        print("Кликните по шару цвета", color)

        while not clicked:
            ret, frame = cam.read()

            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "click " + color, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("camera", frame)
            cv2.waitKey(1)

        blur = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[pos[1], pos[0]]
        lower = [max(0, int(h) - 10), max(0, int(s) - 60), max(0, int(v) - 60)]
        upper = [min(180, int(h) + 10), min(255, int(s) + 60), min(255, int(v) + 60)]
        data[color] = [lower, upper]

    with config_path.open("w") as f:
        json.dump(data, f)

secret = []
for i in range(n):
    secret.append(random.choice(colors))

cam = cv2.VideoCapture(0)
if config_path.exists():
    with config_path.open('r') as file:
        data = json.load(file)
else:
    calibrate(cam)

print("Компьютер загадал")
for c in secret:
    print(c, end=" ")
print()


def mask_for_color(hsv, color):
    lower = np.array(data[color][0])
    upper = np.array(data[color][1])
    return cv2.inRange(hsv, lower, upper)


def sort_balls(balls):
    if n == 3:
        balls.sort(key=lambda b: b[0])
    else:
        balls.sort(key=lambda b: b[1])
        top = balls[:2]
        bottom = balls[2:]
        top.sort(key=lambda b: b[0])
        bottom.sort(key=lambda b: b[0])
        balls = top + bottom
    return balls

score = 0

guessed = False
pause = False
pause_time = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    blur = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    balls = []

    for color in colors:
        mask = mask_for_color(hsv, color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > 700:
                (x, y), r = cv2.minEnclosingCircle(cnt)

                if r > 15:
                    balls.append([int(x), int(y), int(r), color, area])

    balls.sort(key=lambda b: b[4], reverse=True)
    balls = balls[:n]
    balls = sort_balls(balls)
    answer = []

    for b in balls:
        answer.append(b[3])
        cv2.circle(frame, (b[0], b[1]), b[2], (255, 255, 255), 3)

    text = "secret: "

    for c in secret:
        text += c + " "

    cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    if not pause and len(answer) == n and answer == secret:
        pause = True
        pause_time = time.time()
        
        if not guessed:
            print("отгадал")
            guessed = True

    if pause:
        cv2.putText(frame, "OTGADAL", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)

        if time.time() - pause_time > 2:
            secret = []
            for i in range(n):
                secret.append(random.choice(colors))

            print("Компьютер загадал")
            for c in secret:
                print(c, end=" ")
            print()

            pause = False
            guessed = False
    
    cv2.imshow("camera", frame)
    key = cv2.waitKey(1)
    
    if key == ord("n"):
        secret = []
        for i in range(n):
            secret.append(random.choice(colors))
        

    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()