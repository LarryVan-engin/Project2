"""
*******************************************************************************************************************
General Information
********************************************************************************************************************
Project:       DETECT LICENSE PLATES AND TRAFFIC PENALTY INTERGRATION
File:          FULL_MAIN_CODE.py
Description:   This code is intergrated full main code and use for system.

Author:        LARRY PHONG TRUC
Email:         vanphongtruc1808@gmail.com
Created:       05/10/2025
Last Update:   26/10
Version:       1.0

Python:        3.10.11
Dependencies:  - Model YOLOv12 after training for vehicle and plate detect.
               - Model YOLOv12 after training for detect traffic light.
               - Test image/video

Copyright:     (c) 2025 IOE INNOVATION Team
License:       [LICENSE_TYPE]

Notes:         - Just for test, may use for module
*******************************************************************************************************************
"""

#######################################################################################################################
# Imports
#######################################################################################################################
# Standard library imports
import os
import cv2
from ultralytics import YOLO
from PIL import Image

# Third party imports
from preprocessing import preprocess_plate  # import code xử lý hình ảnh preprocessing


#######################################################################################################################
# Constants and Configuration
#######################################################################################################################
# Model YOLO constants
YOLO_VEHICLE_PATH = r"D:\VSCode\DCLP\main_code\runs\detect\train(est.pt"

#INPUT
INPUT_FOLDER = r"D:\VSCode\DCLP\dataset_train\test_image"

#OUTPUT
RESULT_VEHICLE_DETECT = r"D:\VSCode\DCLP\main_code\result\vehicle_detect"
RESULT_PLATE_DETECT = r"D:\VSCode\DCLP\main_code\result\plate_detect"
RESULT_CROPPED_PLATE = r"D:\VSCode\DCLP\main_code\result\crop_plate"
RESULT_AFTER_PREPROCCES = r"D:\VSCode\DCLP\main_code\result\after_preprocess"


#######################################################################################################################
# Main Classes
#######################################################################################################################
def load_model(YOLO_VEHICLE_PATH: str):
    """Load YOLO model từ file weight"""
    model = YOLO(YOLO_VEHICLE_PATH)
    return model


def detect_plate(model, image_path: str):
    """Chạy detect trên ảnh và trả về kết quả"""
    results = model(image_path)
    return results


def plate_detection(results, RESULT_PLATE_DETECT: str):
    """Vẽ khung detect, lưu ảnh kết quả"""
    os.makedirs(os.path.dirname(RESULT_PLATE_DETECT), exist_ok=True)
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save(RESULT_PLATE_DETECT)
    print(f"Đã lưu ảnh detect: {RESULT_PLATE_DETECT}")


def crop_plate_and_preprocess(results, image_path: str, RESULT_CROPPED_PLATE: str, RESULT_AFTER_PREPROCESS: str):
    """Crop vùng biển số đầu tiên phát hiện, lưu ảnh gốc và ảnh đã tiền xử lý"""
    image = cv2.imread(image_path)
    os.makedirs(RESULT_CROPPED_PLATE, exist_ok=True)
    os.makedirs(RESULT_AFTER_PREPROCESS, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if results and results[0].boxes:
        for i, box in enumerate(results[0].boxes.xyxy):
            bbox = box.cpu().numpy()
            x1, y1, x2, y2 = bbox.astype(int)

            # Giới hạn toạ độ trong ảnh
            h, w, _ = image.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cropped_plate = image[y1:y2, x1:x2]

            cropped_name = f"{base_name}_plate{i+1}.jpg"
            cropped_path = os.path.join(RESULT_CROPPED_PLATE, cropped_name)
            cv2.imwrite(cropped_path, cropped_plate)
            print(f"Đã lưu biển số gốc: {cropped_path}")

            processed_name = f"{base_name}_plate{i+1}_proc.jpg"
            processed_path = os.path.join(RESULT_AFTER_PREPROCESS, processed_name)
            preprocess_plate(cropped_path, processed_path)
            print(f"Đã lưu biển số sau preprocessing: {processed_path}")
    else:
        print(f"Không phát hiện biển số trong ảnh {base_name}")


def main_function():
    """
    lay data tu cac ham con

    """
    model = load_model(YOLO_VEHICLE_PATH)

    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            
            # Duong dan toi hinh anh/video
            image_path = os.path.join(INPUT_FOLDER, filename)
            detect_save_path = os.path.join(RESULT_VEHICLE_DETECT, f"detect_{filename}")

            print(f"\nĐang xử lý: {filename}")
            results = detect_plate(model, image_path)
            plate_detection(results, detect_save_path)
            crop_plate_and_preprocess(results, image_path, RESULT_CROPPED_PLATE, RESULT_AFTER_PREPROCCES)


#######################################################################################################################
# Main Execution
#######################################################################################################################
if __name__ == "__main__":
    main_function()

# End of File