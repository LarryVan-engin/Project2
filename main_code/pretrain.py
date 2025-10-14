
###Nên sử dụng Colab để train thay vì train trực tiếp trên máy
from ultralytics import YOLO
import os, shutil

# Đường dẫn thư mục "runs"
runs_dir = "/content/Project2/main_code/runs/detect"
train_dir = os.path.join(runs_dir, "train")

# Xoá thư mục train cũ (nếu có)
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
    print(f"Deleted folder: {train_dir}")

# Main
if __name__ == "__main__":
    # Load model YOLOv12
    model = YOLO("yolo12n.pt")

    # Train
    results = model.train(
        data="/content/Project2/big_dataset/my_dataset.yaml",  #link to dataset
        epochs=30,
        imgsz=640,      #Giảm về 512 nếu train trực tiếp trên laptop
        batch=4,        # có thể tăng batch size vì GPU Colab mạnh hơn
        workers=2,
        device=0,
        cache=True
    )