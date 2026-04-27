import yaml
from ultralytics import YOLO

# Corrige caminhos do yaml
yaml_path = "prototipo/modelos/treino/data.yaml"
with open(yaml_path) as f:
    data = yaml.safe_load(f)

import os
base = os.path.abspath("dataset")
data['train'] = base + "/train/images"
data['val']   = base + "/valid/images"
data['test']  = base + "/test/images"

with open(yaml_path, "w") as f:
    yaml.dump(data, f)

# Treina
model = YOLO("yolov8n-seg.pt")
model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,
    workers=4,
    device=0  # usa GPU
)

print("Modelo salvo em: runs/segment/train/weights/best.pt")