# Webcam Hand Tracker

HandFlow is a Python webcam demo that detects hands in real time and draws MediaPipe hand landmarks directly on the camera feed.

It uses:
- OpenCV for webcam capture and rendering
- MediaPipe Tasks for hand landmark detection
- The included `hand_landmarker.task` model file

## Features

- Real-time webcam hand tracking
- Supports up to two hands
- Draws hand landmark points
- Draws hand skeleton connections
- Mirrored camera preview for a natural webcam view
- Press `q` to quit

---

## How It Works

`main.py` opens your default webcam with OpenCV, flips each frame horizontally, and converts it into a MediaPipe image.

The MediaPipe `HandLandmarker` runs in video mode using `hand_landmarker.task`. For each detected hand, the script draws:

- Purple lines between connected hand landmarks
- White circles on each landmark point

---

## Controls

| Input | Action |
|---|---|
| Show one or two hands | Draw tracked hand landmarks |
| Press `q` | Quit the application |

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Nullit13/handflow.git
cd handflow
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure `hand_landmarker.task` is in the project directory.

The current script expects this model file beside `main.py`:

```text
hand_landmarker.task
```

4. Run the application:

```bash
python main.py
```

---

## Configuration

You can modify these values in `main.py`:

```python
cap = cv2.VideoCapture(0)                 # Webcam index
model_asset_path = "hand_landmarker.task" # MediaPipe model path
num_hands = 2                             # Maximum hands to track
min_hand_detection_confidence = 0.5       # Detection threshold
min_tracking_confidence = 0.7             # Tracking threshold
```

---

## Notes

- Ensure your webcam is connected and accessible.
- Good lighting improves hand detection accuracy.
- If you have multiple cameras, change `cv2.VideoCapture(0)` to another camera index, such as `1`.
- The script does not currently control the mouse or trigger gesture actions.

---

## Future Improvements

- Gesture recognition
- Mouse control mode
- Gesture customization
- Scroll gesture support
- On-screen labels for detected handedness

---

## License

This project is open-source and free to use.
