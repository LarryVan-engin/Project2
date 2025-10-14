import cv2
import os

def preprocess_plate(input_path, output_path, target_w=200):
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Up-sample để nét chữ dày hơn tương đối
    h, w = gray.shape

    if w < target_w:
        scale = target_w / w
        gray = cv2.resize(gray, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Lọc nhẹ giữ cạnh: bilateral
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)

    # CLAHE để tăng tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Adaptive threshold (thử THRESH_BINARY_INV nếu nền sáng)
    binary = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, # hoặc THRESH_BINARY
                               blockSize=11, C=2)

    # Morphological closing nhỏ để nối nét, tránh opening mạnh
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    #Đảo ngược màu ảnh viền
    proc = cv2.bitwise_not(binary)
    proc = proc[10:-10, 10:-10] 

    #Resize ảnh để OCR dễ đọc hơn
    h, w = proc.shape
    scale = 300 / w
    proc = cv2.resize(proc, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Optionally resize back to moderate size or keep as is
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, proc)
