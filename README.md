Face Recognition with Siamese Network and MTCNN
Giới thiệu
Dự án này sử dụng mạng Siamese kết hợp với MTCNN để thực hiện nhận diện khuôn mặt.

-MTCNN: Phát hiện và căn chỉnh khuôn mặt từ hình ảnh hoặc webcam.
-Siamese Network: So sánh hai khuôn mặt và xác định xem chúng có thuộc về cùng một người không.
-Dự án được thiết kế để hoạt động trên dữ liệu tùy chỉnh với cấu trúc có tổ chức.

--Các tính năng chính--
-Phát hiện khuôn mặt: Sử dụng MTCNN để tìm và căn chỉnh khuôn mặt từ ảnh hoặc video.
-Nhận diện khuôn mặt: Dựa trên mạng Siamese để so sánh khuôn mặt đầu vào với cơ sở dữ liệu đã lưu.
-Hỗ trợ thời gian thực: Nhận diện khuôn mặt qua webcam.
-Huấn luyện tùy chỉnh: Huấn luyện mô hình trên dữ liệu của riêng bạn.


--Hướng dẫn sử dụng--
1. Cài đặt môi trường

Cài đặt Python >= 3.8.
Cài đặt các thư viện cần thiết:
pip install -r requirements.txt

2. Chuẩn bị dữ liệu
Lưu ảnh gốc vào thư mục data/raw/, mỗi người một thư mục riêng:

data/raw/
├── person_1/
│   ├── image1.jpg
│   ├── image2.jpg
├── person_2/
    ├── image1.jpg
    ├── image2.jpg

Chạy notebook data_preparation.ipynb để phát hiện khuôn mặt và tạo dữ liệu cặp (positive/negative).

3. Huấn luyện mô hình

Chạy notebook training_siamese.ipynb để huấn luyện mô hình Siamese.

Mô hình đã huấn luyện sẽ được lưu tại models/siamese_model.h5.

4. Nhận diện khuôn mặt qua webcam

Chạy tệp webcam_recognition.py:
python app/webcam_recognition.py

Yêu cầu hệ thống
Python >= 3.8
Thư viện:
TensorFlow
OpenCV
MTCNN
NumPy
Cài đặt đầy đủ thông qua tệp requirements.txt.

