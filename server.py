from flask import Flask, render_template, Response
import cv2
from keras.models import load_model
import numpy as np

app = Flask(__name__)

model = load_model('modelov2.h5')

img_size = 64
frame_skip = 2

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        frame_count += 1
        if frame_count % frame_skip == 0:
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
            rostros = detector.detectMultiScale(frame, 1.3, 5)
            for (x, y, w, h) in rostros:
                rostro = frame[y:y+h, x:x+w]
                rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
                rostro = cv2.resize(rostro, (img_size, img_size))
                img = np.array(rostro).reshape(-1, img_size, img_size, 1)
                img = img / 255.0
                prediction = model.predict(img)
                classes = ['crescencio', 'strange']
                predicted_class = classes[np.argmax(prediction)]
                print(f'La clase predicha es: {predicted_class}')
                umbral = 0.999
                if prediction[0][0] < umbral:
                    predicted_class = "desconocido"
                if predicted_class == "crescencio":
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, predicted_class, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run()