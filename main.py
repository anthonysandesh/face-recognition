from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import torch
import numpy as np

model = YOLO("best.pt")

results = model.predict(source=0, show=True,save=True, save_txt=True, save_conf=True, conf=0.4)
print(results)
