
import os
from ultralytics import YOLO
from PIL import Image

#Load trained model YOLOv11 after epoched - choose the best.pt
model = YOLO(r"D:\VSCode/DCLP\main_code\runs/detect\train2/weights/best.pt")

result = model('D:\VSCode\DCLP\dataset_train/test_image/4.jpg')

#Print the result to screen
for r in result:
    print(r.boxes)
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1]) #RGB PIL image
    im.show()
    save_path = os.path.join('D:/VSCode/DCLP/dataset_train/result', 'result.jpg')
    im.save(save_path)

