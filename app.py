from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import threading
import time

app = Flask(__name__)

# ======================
# LOAD MODELS
# ======================
weapon_model = YOLO("weapon_yolo.pt")
emotion_model = tf.keras.models.load_model("emotion_model.keras")

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ======================
# GLOBAL STATE
# ======================
stats = {
    "suspicious_count": 0,
    "details": [],
    "current_mood": "unknown",
    "alert_level": "NORMAL"  # NORMAL | WARNING | ALERT
}

camera_running = False
cap = None
lock = threading.Lock()
last_mood_time = 0

# ======================
# EMOTION FUNCTION
# ======================
def predict_emotion(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    gray = gray / 255.0
    gray = gray.reshape(1, 48, 48, 1)
    pred = emotion_model.predict(gray, verbose=0)
    return EMOTIONS[int(np.argmax(pred))]

# ======================
# CENTER FACE (FOR TESTING MOOD)
# ======================
def get_monitor_face(frame):
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 3
    size = min(h, w) // 4
    return frame[cy-size:cy+size, cx-size:cx+size]

# ======================
# VIDEO STREAM
# ======================
def generate_frames():
    global cap, camera_running, last_mood_time

    while True:
        if not camera_running or cap is None:
            time.sleep(0.2)
            continue

        success, frame = cap.read()
        if not success:
            continue

        # -------- CONTINUOUS MOOD CHECK (every 2 sec) --------
        if time.time() - last_mood_time > 2:
            try:
                face = get_monitor_face(frame)
                mood = predict_emotion(face)
            except:
                mood = "unknown"

            with lock:
                stats["current_mood"] = mood

            last_mood_time = time.time()

        # -------- YOLO DETECTION --------
        detections = weapon_model(frame, conf=0.4)[0]
        suspicious = []
        alert = "NORMAL"

        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            emotion = "unknown"
            if crop.size > 0:
                emotion = predict_emotion(crop)

            danger = emotion in ["angry", "fear"]

            if danger:
                alert = "ALERT"
            elif alert != "ALERT":
                alert = "WARNING"

            color = (0, 0, 255) if danger else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, emotion.upper(), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            suspicious.append({
                "emotion": emotion,
                "danger": danger
            })

        with lock:
            stats["suspicious_count"] = len(suspicious)
            stats["details"] = suspicious
            stats["alert_level"] = alert

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

# ======================
# ROUTES
# ======================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def get_stats():
    with lock:
        return jsonify(stats)

@app.route("/start_camera", methods=["POST"])
def start_camera():
    global cap, camera_running
    if not camera_running:
        cap = cv2.VideoCapture(0)
        camera_running = True
    return jsonify({"status": "started"})

@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global cap, camera_running
    camera_running = False
    if cap:
        cap.release()
        cap = None
    return jsonify({"status": "stopped"})

# ======================
# START
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0",debug=True, threaded=True,port=port)
