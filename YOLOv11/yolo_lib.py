from ultralytics import YOLO
import numpy as np
import cv2
class MyYOLO():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    def update(self,image:np.ndarray,content:dict):
        results=self.model(image,show=True)
        # results.print()