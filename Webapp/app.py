from flask import Flask, render_template, request
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import time
import os
from werkzeug.utils import secure_filename

# Install model recognization face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model('model_best.keras')

# Recognization face from image
def detect_and_preprocess_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image into grayscale
    # Find face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # Check face
    if len(faces) == 0:
        print("Không phát hiện khuôn mặt.")
        return None
    # Cut first face
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    # Resize face (48x48)
    face_resized = cv2.resize(face, (48, 48))
    cut_face_image = cv2.fastNlMeansDenoising(face_resized, None, 10, 7, 21)
    return cut_face_image

# Dự đoán cảm xúc
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

def predict_emotion(face_image):
    # Chuẩn hóa ảnh trước khi đưa vào mô hình
    face_image = face_image / 255.0  # Scale pixel về [0, 1]
    face_image = np.expand_dims(face_image, axis=(0, -1))  # Thêm batch size và channel

    # Dự đoán
    predictions = model.predict(face_image)
    predicted_emotion = emotion_labels[np.argmax(predictions)]
    return predicted_emotion

# Khởi tạo Flask app
app = Flask(__name__)
# Tạo thư mục uploads nếu chưa có
UPLOAD_FOLDER = './static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
# Cấu hình Flask để biết thư mục lưu ảnh
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
# Kiểm tra định dạng ảnh hợp lệ
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
CORS(app)  # Cho phép tất cả các nguồn tải tài nguyên
@app.route('/', methods=['GET', 'POST'])
def index():
    emotion = None
    image_url = None  # Khởi tạo image_url là None
    if request.method == 'POST':
        file = request.files['file']  # Đảm bảo tên key là 'file'
        if file:
            filename = secure_filename(file.filename)
            if allowed_file(file.filename):
                timestamp_filename = str(int(time.time())) + os.path.splitext(file.filename)[1]
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], timestamp_filename)
                file.save(file_path)
                image_url = str( 'uploads/' + timestamp_filename)
                print(image_url)
                # Đọc ảnh và nhận diện cảm xúc
                img = cv2.imread(file_path)
                if img is not None:
                    face_image = detect_and_preprocess_face(img)
                    if face_image is not None:
                        emotion = predict_emotion(face_image)
            else:
                print(f"File format not allowed: {file.filename}")
        else:
            print("No file received")
    return render_template('index.html',image_url=image_url, emotion=emotion, upload=bool(emotion))
if __name__ == "__main__":
    app.run(debug=True)