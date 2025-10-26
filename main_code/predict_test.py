"""
*******************************************************************************************************************
General Information
********************************************************************************************************************
Project:       DETECT LICENSE PLATES AND TRAFFIC PENALTY INTERGRATION
File:          PREDICT_TEST.py
Description:   This code is used to test prediction with model YOLO after training.

Author:        LARRY PHONG TRUC
Email:         vanphongtruc1808@gmail.com
Created:       05/10/2025
Last Update:   26/10
Version:       1.0

Python:        3.10.11
Dependencies:  - Model YOLOv12 after training
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
from ultralytics import YOLO
from PIL import Image


#######################################################################################################################
# Constants and Configuration
#######################################################################################################################
#Constants path
YOLO_PATH = r"./runs/detect/train6/weights/best.pt"
IMAGE_TEST_PATH = "D:\VSCode\DCLP/big_dataset/test\image.png"
SAVE_PATH = "D:\VSCode\DCLP\main_code/result/vehicle_detect"

# Configuration
#Load trained model YOLOv11 after epoched - choose the best.pt
model = YOLO(YOLO_PATH)
result = model(IMAGE_TEST_PATH)


#######################################################################################################################
# Main Classes
#######################################################################################################################
#Print the result to screen
def main_function():
    for r in result:
        print(r.boxes)
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1]) #RGB PIL image
        im.show()
        save_path = os.path.join(SAVE_PATH, 'result.jpg')
        im.save(save_path)


#######################################################################################################################
# Main Execution
#######################################################################################################################
if __name__ == "__main__":
    main_function()

# End of File