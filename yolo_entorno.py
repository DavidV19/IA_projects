# Instalamos las librerías necesarias
#%pip install opencv-python ultralytics

# Importamos las librerías necesarias
from ultralytics import YOLO
import cv2
import math

# Importamos las librerías necesarias
from ultralytics import YOLO
import cv2
import math

# Revisamos las clases que puede detectar el modelo
#mis clases 
# Cargamos el modelo YOLO pre-entrenado en COCO dataset
model = YOLO("yolo-Weights/yolov8n.pt")

classNames =  ["tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", 
              ]

# Configuramos la captura de video desde la cámara
captura = cv2.VideoCapture(0) # Se abre la cámara por defecto

# Establecemos el ancho y alto de la imagen
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Ancho de la imagen
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Alto de la imagen

# Iniciamos un bucle para procesar los fotogramas de la cámara
while True:
    success, img = captura.read() # Capturamos un fotograma

    # Realizamos la detección de objetos en la imagen capturada (usando el modelo de YOLO pre-entrenado que cargamos anteriormente)
    results = model(img, stream=True)

   # Procesamos los resultados de la detección
    for r in results:
        boxes = r.boxes

        # Iteramos sobre las cajas delimitadoras detectadas
        for box in boxes:
            # Obtenemos las coordenadas de la caja delimitadora
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convertimos a valores enteros

            # Dibujamos la caja delimitadora en la imagen
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)

            # Obtenemos la confianza de la detección
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # Obtenemos el nombre de la clase detectada
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # Mostramos el nombre de la clase junto a la caja delimitadora
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0) # Color: Azul (formato BGR)
            thickness = 1
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    # Mostramos la imagen con las detecciones
    cv2.imshow('Webcam', img)

    # Salimos del bucle si se presiona la tecla 'q'
    #if cv2.waitKey(1) == ord('q'):
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberamos la cámara y cerramos todas las ventanas
captura.release()
cv2.destroyAllWindows()