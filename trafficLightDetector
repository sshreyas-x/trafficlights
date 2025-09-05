import cv2 as cv
import numpy as np

# Detection Function
def detect_traffic_lights(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # HSV ranges
    lower_red1 = np.array([0, 120, 120])
    upper_red1 = np.array([8, 255, 255])
    lower_red2 = np.array([172, 120, 120])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv.inRange(hsv, lower_red1, upper_red1) | cv.inRange(hsv, lower_red2, upper_red2)

    lower_yellow = np.array([8, 120, 150])
    upper_yellow = np.array([38, 255, 255])
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)

    lower_green = np.array([35, 60, 60])
    upper_green = np.array([95, 255, 255])
    mask_green = cv.inRange(hsv, lower_green, upper_green)

    def detect_and_draw(mask, color_name, color_bgr):
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < 80:  # filter small noise
                continue

            perimeter = cv.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.6:  # keep mostly circular
                continue

            x, y, w, h = cv.boundingRect(cnt)
            roi = frame[y:y+h, x:x+w]
            gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            gray_roi = cv.medianBlur(gray_roi, 5)

            # Circle check using HoughCircles
            circles = cv.HoughCircles(
                gray_roi, cv.HOUGH_GRADIENT, dp=1, minDist=10,
                param1=50, param2=15, minRadius=5, maxRadius=max(w, h)//2
            )

            if circles is not None:
                cv.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
                cv.putText(frame, color_name, (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

    detect_and_draw(mask_red, "RED", (0, 0, 255))
    detect_and_draw(mask_yellow, "YELLOW", (0, 255, 255))
    detect_and_draw(mask_green, "GREEN", (0, 255, 0))

    return frame


# MODE SELECTION
print("Select Mode:")
print("1 - Image")
print("2 - Video File")
print("3 - Camera")
mode = input("Enter mode number: ")

if mode == "1":
    path = input("Enter image path: ")
    frame = cv.imread(path)
    if frame is None:
        print("Error: Image not found!")
    else:
        result = detect_traffic_lights(frame)
        cv.imshow("Traffic Light Detection", result)
        cv.waitKey(0)
        cv.destroyAllWindows()

elif mode == "2":
    path = input("Enter video path: ")
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = detect_traffic_lights(frame)
            cv.imshow("Traffic Light Detection", result)
            if cv.waitKey(1) & 0xFF == 27:  # ESC
                break
        cap.release()
        cv.destroyAllWindows()

elif mode == "3":
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access camera.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = detect_traffic_lights(frame)
            cv.imshow("Traffic Light Detection", result)
            if cv.waitKey(1) & 0xFF == 27:  # ESC
                break
        cap.release()
        cv.destroyAllWindows()

else:
    print("Invalid mode!")

