from ultralytics import YOLO

###Tiếp tục train sau dựa trên file pretrain trước, giữ nguyên dữ liệu đã train và nạp thêm
###Ưu tiên train biển số vàng hoặc biển số xanh

model = YOLO("main_code\runs\detect\train2\weights\last.pt") #link to 'last.pt'

model.train(
    data="/content/Project2/big_dataset/my_dataset.yaml",
    epochs=60,       # train thêm 30 epoch nữa
    imgsz=640,
    batch=4,         #Batch = 8 is use GPU T4 on Colab
    workers=2,
    device=0,
    resume=True
)