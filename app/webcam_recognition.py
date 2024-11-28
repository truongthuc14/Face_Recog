import cv2
import numpy as np
from keras.models import load_model
from mtcnn import MTCNN
from scipy.spatial.distance import cosine

# Đường dẫn đến mô hình và cơ sở dữ liệu
MODEL_PATH = "models/siamese_model.h5"
DATABASE_PATH = "data/database_embeddings.npy"

# Tải mô hình đã huấn luyện và cơ sở dữ liệu
print("Đang tải mô hình và cơ sở dữ liệu...")
siamese_model = load_model(MODEL_PATH)
base_model = siamese_model.layers[2]  # Lớp trích xuất đặc trưng
database = np.load(DATABASE_PATH, allow_pickle=True).item()

# Chuẩn hóa embeddings trong cơ sở dữ liệu
for person_name in database:
    database[person_name] = [
        embedding / np.linalg.norm(embedding) for embedding in database[person_name]
    ]
print("Tải thành công! Cơ sở dữ liệu đã chuẩn hóa.")

# Khởi tạo MTCNN để phát hiện khuôn mặt
detector = MTCNN()

# Hàm chuẩn hóa ảnh
def preprocess_image(image, target_size=(160, 160)):
    """
    Chuẩn hóa ảnh để đưa vào mô hình.
    Args:
        image (np.ndarray): Khuôn mặt đã cắt.
        target_size (tuple): Kích thước đích.
    Returns:
        np.ndarray: Ảnh đã chuẩn hóa.
    """
    image = cv2.resize(image, target_size)
    image = np.array(image, dtype=np.float32) / 255.0  # Chuẩn hóa pixel về [0, 1]
    return np.expand_dims(image, axis=0)

# Hàm dự đoán người
def predict_person(face_crop, database, base_model, threshold=0.5):
    """
    Nhận diện người từ khuôn mặt.
    Args:
        face_crop (np.ndarray): Khuôn mặt đã cắt.
        database (dict): Cơ sở dữ liệu embeddings.
        base_model (Model): Mô hình trích xuất đặc trưng.
        threshold (float): Ngưỡng để quyết định nhận diện.
    Returns:
        str: Tên người hoặc "Unknown".
        float: Khoảng cách nhỏ nhất.
    """
    # Chuẩn hóa ảnh và trích xuất embedding
    face_embedding = base_model.predict(preprocess_image(face_crop)).flatten()
    face_embedding = face_embedding / np.linalg.norm(face_embedding)  # Chuẩn hóa embedding

    closest_person = None
    min_distance = float('inf')

    for person_name, embeddings in database.items():
        for db_embedding in embeddings:
            distance = cosine(face_embedding, db_embedding)  # Khoảng cách Cosine
            if distance < min_distance:
                min_distance = distance
                closest_person = person_name

    # Kiểm tra ngưỡng
    if min_distance < threshold:
        return closest_person, min_distance
    else:
        return "Unknown", min_distance

# Nhận diện qua camera
def recognize_from_camera(threshold=0.5):
    """
    Nhận diện khuôn mặt từ camera.
    """
    cap = cv2.VideoCapture(0)  # Mở camera
    if not cap.isOpened():
        print("Không thể mở camera!")
        return

    print("Bắt đầu nhận diện... Nhấn 'q' để thoát.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc từ camera!")
            break

        # Lật khung hình theo chiều ngang
        frame = cv2.flip(frame, 1)

        # Phát hiện khuôn mặt mỗi 10 khung hình
        if frame_count % 20 == 0:
            faces = detector.detect_faces(frame)

        frame_count += 1

        for face in faces:
            x, y, w, h = face['box']
            confidence = face['confidence']

            # Kiểm tra vùng cắt hợp lệ
            x, y = max(0, x), max(0, y)
            h, w = max(0, h), max(0, w)

            if confidence > 0.9:  # Chỉ xử lý khuôn mặt có độ tin cậy cao
                # Cắt khuôn mặt
                face_crop = frame[y:y+h, x:x+w]

                # Nhận diện khuôn mặt
                person, distance = predict_person(face_crop, database, base_model, threshold)

                # Vẽ khung và hiển thị tên người
                color = (0, 255, 0) if person != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{person} ({distance:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Hiển thị luồng camera
        cv2.imshow("Webcam Face Recognition", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_from_camera(threshold=0.5)
