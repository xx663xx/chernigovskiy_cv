import cv2

img = cv2.imread("balls_and_rects.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
circles = {}
rects = {}

for c in contours:
    m = cv2.moments(c)
    if m["m00"] == 0:
        continue

    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    b, g, r = img[cy, cx]
    color = (int(r), int(g), int(b))
    p = cv2.arcLength(c, True)
    a = cv2.approxPolyDP(c, 0.04 * p, True)

    if len(a) == 4:
        if color not in rects:
            rects[color] = 0
        rects[color] += 1
    else:
        if color not in circles:
            circles[color] = 0
        circles[color] += 1

total = len(contours)

print(f"Всего {total} фигур")
print(f"{sum(rects.values())} прямоугольников")
print(f"{sum(circles.values())} кругов")
print("\nПрямоугольники по оттенкам")

for color in sorted(rects):
    print(f"RGB {color} - {rects[color]}")

print("\n\nКруги по оттенкам")

for color in sorted(circles):
    print(f"RGB {color} - {circles[color]}")