import os
import cv2
from ultralytics import YOLO
from PIL import Image
from preprocessing import preprocess_plate  # import code xử lý hình ảnh preprocessing


def load_model(weight_path: str):
    """Load YOLO model từ file weight"""
    model = YOLO(weight_path)
    return model


def detect_plate(model, image_path: str):
    """Chạy detect trên ảnh và trả về kết quả"""
    results = model(image_path)
    return results


def show_detection(results, save_path: str):
    """Vẽ khung detect, lưu ảnh kết quả"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save(save_path)
    print(f"Đã lưu ảnh detect: {save_path}")


def crop_plate(results, image_path: str, crop_folder: str, preprocess_folder: str):
    """Crop vùng biển số đầu tiên phát hiện, lưu ảnh gốc và ảnh đã tiền xử lý"""
    image = cv2.imread(image_path)
    os.makedirs(crop_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if results and results[0].boxes:
        for i, box in enumerate(results[0].boxes.xyxy):
            bbox = box.cpu().numpy()
            x1, y1, x2, y2 = bbox.astype(int)

            # ✅ Giới hạn toạ độ trong ảnh
            h, w, _ = image.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cropped_plate = image[y1:y2, x1:x2]

            cropped_name = f"{base_name}_plate{i+1}.jpg"
            cropped_path = os.path.join(crop_folder, cropped_name)
            cv2.imwrite(cropped_path, cropped_plate)
            print(f"Đã lưu biển số gốc: {cropped_path}")

            processed_name = f"{base_name}_plate{i+1}_proc.jpg"
            processed_path = os.path.join(preprocess_folder, processed_name)
            preprocess_plate(cropped_path, processed_path)
            print(f"Đã lưu biển số sau preprocessing: {processed_path}")
    else:
        print(f"Không phát hiện biển số trong ảnh {base_name}")


def main():
    weight_path = r"D:\VSCode\DCLP\main_code\runs\detect\train2\weights\best.pt"
    input_folder = r"D:\VSCode\DCLP\dataset_train\test_image"

    detect_folder = r"D:\VSCode\DCLP\main_code\result\plate_detect"
    crop_folder = r"D:\VSCode\DCLP\main_code\result\crop_plate"
    after_preprocess = r"D:\VSCode\DCLP\main_code\result\after_preprocess"

    model = load_model(weight_path)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            detect_save_path = os.path.join(detect_folder, f"detect_{filename}")

            print(f"\nĐang xử lý: {filename}")
            results = detect_plate(model, image_path)
            show_detection(results, detect_save_path)
            crop_plate(results, image_path, crop_folder, after_preprocess)


if __name__ == "__main__":
    main()
