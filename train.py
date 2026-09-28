
from ultralytics import YOLO
import torch
torch.set_num_threads(2)
model = YOLO("yolov8n.pt")
model.train(data=r"C:\Users\aysen\Desktop\veri\YOLODataset\dataset.yaml", epochs=50, imgsz=640, batch=8, device="cpu", patience=5, optimizer="SGD", lr0=0.001)