from ultralytics import YOLO
model = YOLO(r"C:\Users\aysen\Desktop\proje\runs\detect\train-7\weights\best.pt")
model.predict(source=0, imgsz=640, show=True, save=True, conf=0.7)