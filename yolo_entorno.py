# Instalamos las librerías necesarias
# %pip install opencv-python ultralytics

# Importamos las librerías necesarias
from ultralytics import YOLO
import cv2
import math

# Cargamos el modelo YOLO pre-entrenado en COCO dataset
model = YOLO("yolo-Weights/yolov8n.pt")

# Lista completa de clases del modelo
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Clases que SÍ queremos mostrar (enonctrar objetos con direcciones IP y Ssitema operativo)
clases_filtradas = ["tvmonitor", "cell phone", "laptop"]

# Configuramos la captura de video desde la cámara
captura = cv2.VideoCapture(0)  # Cámara por defecto
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Bucle para procesamiento en tiempo real
while True:
    success, img = captura.read()  # Leemos un frame

    # Detección de objetos
    results = model(img, stream=True)

    # Procesamos cada resultado
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordenadas de la caja
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confianza de detección
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Clase detectada
            cls = int(box.cls[0])
            nombre_clase = classNames[cls]

            # Filtramos solo las clases deseadas
            if nombre_clase in clases_filtradas:
                print("Confidence --->", confidence)
                print("Class name -->", nombre_clase)

                # Dibujamos la caja y el texto
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)  # Azul
                thickness = 1
                cv2.putText(img, nombre_clase, org, font, fontScale, color, thickness)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)

    # Mostramos el resultado en ventana
    cv2.imshow('Webcam', img)

    # Presiona 'q' para salir
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberamos recursos
captura.release()
cv2.destroyAllWindows()
