import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Você não tem uma camera compativel ou não permitiu o uso das cameras.")
    exit()

model = YOLO("prototipo/modelos/meu_modelov2.pt")

#region posição e informação
def pos(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    area_obj = (x2 - x1) * (y2 - y1)
    area_frame = frame_w * frame_h

    if cx < frame_w * 0.33:
        horizontal = "saindo"
    elif cx < frame_w * 0.66:
        horizontal = "Centro"
    else:
        horizontal = "entrando"

    if cy < frame_h * 0.33:
        vertical = "Cima"
    elif cy < frame_h * 0.66:
        vertical = "Meio"
    else:
        vertical = "Baixo"

    proporcao = area_obj / area_frame
    if proporcao > 0.3:
        distancia = "Muito perto"
    elif proporcao > 0.1:
        distancia = "Perto"
    elif proporcao > 0.03:
        distancia = "Medio"
    else:
        distancia = "Longe"

    return f"{horizontal} | {vertical} | {distancia}"
#endregion

#region camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro: não recebeu frame...")
        break

    h, w = frame.shape[:2]
    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
    annotated = results[0].plot(masks=False)

    boxes = results[0].boxes
    if len(boxes) == 0:
        cv2.putText(annotated, "Nenhum objeto", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        for i, r in enumerate(boxes):
            classe = model.names[int(r.cls)]
            confianca = float(r.conf)
            box = r.xyxy[0].tolist()
            posicao = pos(box, w, h)
            texto = f"{classe} {confianca:.0%} | {posicao}"
            cv2.putText(annotated, texto, (10, 30 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(texto)

    # imshow DEPOIS do putText, waitKey obrigatório
    cv2.imshow("Camera prototipo com detecção de imgs", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#endregion

cap.release()
cv2.destroyAllWindows()