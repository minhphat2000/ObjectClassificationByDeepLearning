import cv2
import os

# Tạo thư mục 'images' nếu chưa tồn tại
if not os.path.exists("Phat"):
    os.makedirs("Phat")

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Lặp lại việc chụp hình ảnh
counter = 0
while True:
    # Đọc hình ảnh từ webcam
    ret, frame = cap.read()

    # Hiển thị hình ảnh trên màn hình
    cv2.imshow("Webcam", frame)

    # Chờ cho phím 'q' được nhấn để thoát khỏi vòng lặp
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # Lưu hình ảnh vào thư mục 'images'
    if key == ord('s'):
        filename = "cup/cup{}.png".format(counter)
        cv2.imwrite(filename, frame)
        counter += 1

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
