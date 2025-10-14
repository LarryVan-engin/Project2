from ultralytics import YOLO
import cv2

# Load model đã train nhận diện biển số (đường dẫn file weights .pt)
model = YOLO(r"D:\VSCode/DCLP\main_code\runs/detect\train2/weights/best.pt")

# Đọc ảnh cần detect
image = cv2.imread("D:\VSCode\DCLP\dataset_train/test_image/3.jpg")

# Chạy detect trên ảnh
results = model(image)

# Lấy bbox kết quả đầu tiên (nếu phát hiện được)
if results and results[0].boxes:
    # results[0].boxes.xyxy trả về list bbox dạng [x1, y1, x2, y2]
    bbox = results[0].boxes.xyxy[0].cpu().numpy()

    x1, y1, x2, y2 = bbox.astype(int)

    # Crop vùng biển số theo bbox
    cropped_plate = image[y1:y2, x1:x2]

    # Hiển thị ảnh crop vùng biển số
    cv2.imshow("Cropped License Plate", cropped_plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Lưu ảnh crop
    cv2.imwrite("cropped_plate.jpg", cropped_plate)
else:
    print("Không phát hiện biển số trên ảnh")
