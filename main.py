import cv2
import mediapipe as mp
import time
import ctypes
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

WINDOW_NAME = "Hand Tracking"
SCREEN_WIDTH = ctypes.windll.user32.GetSystemMetrics(0)
SCREEN_HEIGHT = ctypes.windll.user32.GetSystemMetrics(1)
DISPLAY_WIDTH = int(SCREEN_WIDTH * 0.9)
DISPLAY_HEIGHT = int(DISPLAY_WIDTH * 9 / 16)

if DISPLAY_HEIGHT > int(SCREEN_HEIGHT * 0.9):
    DISPLAY_HEIGHT = int(SCREEN_HEIGHT * 0.9)
    DISPLAY_WIDTH = int(DISPLAY_HEIGHT * 16 / 9)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

hands = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.7,
    )
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.monotonic() * 1000)
    results = hands.detect_for_video(mp_image, timestamp_ms)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                x1 = int(hand_landmarks[connection.start].x * w)
                y1 = int(hand_landmarks[connection.start].y * h)
                x2 = int(hand_landmarks[connection.end].x * w)
                y2 = int(hand_landmarks[connection.end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
