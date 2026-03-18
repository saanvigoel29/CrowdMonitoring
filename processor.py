
import cv2
import csv
import datetime
import numpy as np
import os
from ultralytics import YOLO


DENSITY_THRESHOLDS = [
    (0, 5, "Low", (0, 255, 0)),       
    (6, 15, "Medium", (0, 255, 255)), 
    (16, 9999, "High", (0, 0, 255))   
]

LOG_CSV = "crowd_data.csv"


class CrowdProcessor:
    def __init__(self):
      
        self.model = YOLO("yolov8n.pt")
        self.fps = 0
        self.heatmap = None

        
        if not os.path.exists(LOG_CSV):
            with open(LOG_CSV, "w", encoding="utf-8", newline="") as f:
                f.write("timestamp,people_count,crowd_level\n")

    def process(self, frame):
       
        results = self.model.predict(frame, classes=[0], verbose=False)

        people_count = 0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                people_count += 1

        
        crowd_level, color = "Unknown", (255, 255, 255)
        for lo, hi, label, c in DENSITY_THRESHOLDS:
            if lo <= people_count <= hi:
                crowd_level, color = label, c
                break

        
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.datetime.now(), people_count, crowd_level])

       
        if self.heatmap is None:
            
            self.heatmap = np.zeros((frame.shape[0], frame.shape[1]), np.float32)

        self.heatmap *= 0.93  

        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
               
                x1 = max(0, x1 - 5)
                y1 = max(0, y1 - 5)
                x2 = min(frame.shape[1], x2 + 5)
                y2 = min(frame.shape[0], y2 + 5)
                self.heatmap[y1:y2, x1:x2] += 3  

        
        heatmap_norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_TURBO)
       

        frame = cv2.addWeighted(frame, 0.35, heatmap_color, 0.65, 0)

        
        cv2.putText(frame,
                    f"People: {people_count} | Crowd: {crowd_level} | FPS: {self.fps}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

        return frame, people_count, crowd_level
