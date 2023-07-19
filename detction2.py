import cv2
import argparse
import csv
from datetime import datetime

from ultralytics import YOLO
import supervision as sv
import numpy as np

model = YOLO("best.pt")

# open webcam
video_capture = cv2.VideoCapture(0)  # Use index 0 for the default webcam

# Create CSV file and write header
csv_file = open("detections.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Date", "Time", "Labels", "Count"])

while True:
    # read frame from webcam
    ret, frame = video_capture.read()

    # detect
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_yolov8(results)
    detections = detections[detections.class_id == 0]

    # annotate
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # Get current date and time
    current_time = datetime.now()
    date = current_time.strftime("%Y-%m-%d")
    time = current_time.strftime("%H:%M:%S")

    # Write detection information to CSV file
    csv_writer.writerow([date, time, ", ".join(labels), len(detections)])

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()

# Close CSV file
csv_file.close()
