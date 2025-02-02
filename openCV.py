import cv2
import numpy as np
from tensorflow.keras.models import load_model


cap = cv2.VideoCapture(0)
model = load_model("C:/Users/pc/Downloads/model_best.keras")  # Thay bằng đường dẫn tới mô hình của bạn
emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprise']

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể kết nối tới camera.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48)) 
        face_normalized = face_resized / 255.0 
        face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))
        prediction = model.predict(face_reshaped, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
