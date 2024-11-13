import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('yolov8n.pt')  # Puedes usar cualquier variante de YOLOv8

# Función para dibujar texto con sombra
def draw_text_with_shadow(image, text, position, font_scale, color, thickness, shadow_color, offset):
    # Dibujar la sombra (offset en x e y)
    cv2.putText(image, text, (position[0] + offset, position[1] + offset), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow_color, thickness + 2)
    # Dibujar el texto principal
    cv2.putText(image, text, position, 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# Inicialización de la captura de video
cap = cv2.VideoCapture(0)  # Captura desde la cámara
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Configurar el tamaño de la ventana para visualización
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Ancho de la cámara
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Alto de la cámara

while True:
    ret, frame = cap.read()  # Leer un frame de la cámara
    if not ret:
        print("Error al leer el frame.")
        break

    # Realizar detección con YOLO
    results = model(frame)  # Detecta objetos en el frame

    # Acceder a las cajas, las clases y las confidencias
    boxes = results[0].boxes  # Obtener las cajas de detección
    for box in boxes:
        # Extracción de las coordenadas de la caja (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Las coordenadas están en formato [x1, y1, x2, y2]
        
        # Extraer el valor de la clase y la confianza
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # Obtener el nombre de la clase
        class_name = model.names[class_id]

        # Dibujar la caja delimitadora y el texto con sombra
        color = (0, 255, 0)  # Color de la caja (verde)
        thickness = 2
        draw_text_with_shadow(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), 
                              font_scale=0.5, color=color, thickness=2, shadow_color=(0, 0, 0), offset=2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)  # Dibujar la caja

    # Mostrar el frame con las detecciones
    cv2.imshow("Detección YOLO", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
