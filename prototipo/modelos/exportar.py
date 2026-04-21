from ultralytics import YOLO

#compativel com android
model = YOLO("prototipo/modelos/meu_modelo.pt")
model.export(format="onnx") 

import onnx

model = onnx.load("prototipo/modelos/meu_modelo.onnx")

# converte para opset 21
from onnx import version_converter
converted_model = version_converter.convert_version(model, 21)

onnx.save(converted_model, "modelo_opset21.onnx")