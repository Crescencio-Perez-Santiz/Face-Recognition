import cv2
import os

dir_imagenes = './Entrenamiento/crescencio'
if not os.path.exists(dir_imagenes):
    os.makedirs(dir_imagenes)

cap = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

contador = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    rostros = detector.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in rostros:
        rostro = frame[y:y+h, x:x+w]

        cv2.imshow('Captura de rostro', rostro)

        cv2.imwrite(os.path.join(dir_imagenes, f'rostro_{contador}.jpg'), rostro)
        contador += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()