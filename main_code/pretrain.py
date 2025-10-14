from ultralytics import YOLO

#Delete the reply file train before epoched
import shutil, os

runs_dir = "runs/detect"
train_dir = os.path.join(runs_dir, "train")

if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
    print(f"Deleted folder: {train_dir}")

#Main code to train
if __name__=="__main__":
    # Load a COCO-pretrained YOLO11n model
    model = YOLO("yolo12n.pt")

    # Train the model on the pretrain example dataset for 10 epochs
    results = model.train(
        data="D:\VSCode\DCLP\dataset_train\my_dataset.yaml", 
        epochs=30, 
        imgsz=512, 
        batch=2, 
        workers=0, 
        device = 0,
        cache=False
        )
