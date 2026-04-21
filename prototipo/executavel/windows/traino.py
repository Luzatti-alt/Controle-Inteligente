from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="modelos/treino/data.yaml",
    epochs=50,
    imgsz=640
)

model = YOLO("runs/detect/train/weights/best.pt")
print("Modelo treinado")