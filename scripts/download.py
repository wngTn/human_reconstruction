import gdown
import wget

SMPL_MODEL_PATH = "external/EasyMocap-master/data/bodymodels/smpl/models"
YOLOV4_NET_PATH = "external/EasyMocap-master/data/models/"
HRNET_PATH = "external/EasyMocap-master/data/models/"

try:
    # smpl model
    gdown.download_folder(
        url="https://drive.google.com/drive/u/0/folders/1gFdZC4quxsAzqGWR-aY7NFEWC_FeFHVy",
        quiet=False,
        output=SMPL_MODEL_PATH,
    )
    gdown.download(
        url="https://drive.google.com/file/d/1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS/view?usp=sharing",
        quiet=False,
        fuzzy=True,
        output=YOLOV4_NET_PATH
    ) 
    wget.download("https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights", 
                  out=YOLOV4_NET_PATH)
except IOError:
    print("There has been an error trying to install the SMPL Models")