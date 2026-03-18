

import os
import cv2
import time
import pandas as pd
from flask import Flask, Response, render_template, jsonify
from processor import CrowdProcessor

VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "0")
try:
    VIDEO_SOURCE = int(VIDEO_SOURCE)
except ValueError:
    pass

app = Flask(__name__)
processor = CrowdProcessor()

last_people_count = 0
last_crowd_level = "Low"


def gen_frames():
    global last_people_count, last_crowd_level

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    last_fps_time = time.time()
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, people_count, crowd_level = processor.process(frame)
        last_people_count = people_count
        last_crowd_level = crowd_level

        
        frames += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            processor.fps = frames
            frames = 0
            last_fps_time = now

       
        ret2, buffer = cv2.imencode(".jpg", annotated)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify({
        "people": last_people_count,
        "crowd_level": last_crowd_level
    })


@app.route("/stats")
def stats():
    return render_template("stats.html")


@app.route("/data")
def data():
    try:
        df = pd.read_csv("crowd_data.csv")
        timestamps = df["timestamp"].tolist()[-50:]
        counts = df["people_count"].tolist()[-50:]
        return jsonify({"timestamps": timestamps, "counts": counts})
    except Exception:
        return jsonify({"timestamps": [], "counts": []})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


