import cv2
from ultralytics import YOLO
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Você não tem uma camera compativel ou não permitiu o uso das cameras.")
    exit()

model = YOLO("prototipo/modelos/meu_modelov2.pt")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: não recebeu frame...")
        break
    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)

    annotated = results[0].plot()
    cv2.imshow("Camera prototipo com detecção de imgs",annotated)

    if len(results[0].boxes) == 0:
        print("Nenhum objeto detectado")
    if len(results[0].boxes) != 0:
        for r in results[0].boxes:
            classe = model.names[int(r.cls)]
            confianca = float(r.conf)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()