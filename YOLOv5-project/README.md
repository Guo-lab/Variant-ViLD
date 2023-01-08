# Download model

```sh
(zeroShot) GSQdeMacBook-Pro:[8]-YOLOv5-project gsq$ python utilize_YOLOv5.py 
Using cache found in /Users/gsq/.cache/torch/hub/ultralytics_yolov5_master
YOLOv5 🚀 2022-11-29 Python-3.9.12 torch-1.7.1 CPU
Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt to yolov5s.pt...
ERROR: Downloaded file 'yolov5s.pt' does not exist or size is < min_bytes=100000.0
yolov5s.pt missing, try downloading from https://github.com/ultralytics/yolov5/releases/v7.0 or https://drive.google.com/drive/folders/1EFQTEUeXWSFww0luse2jB9M1QNZQGwNl
Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt to yolov5s.pt...
###################################################################################################################################################################################################################################################### 100.0%
```

Next, Utilize model to do inference.
<br><br>

# Transfer Learning
Download the dataset: `oi_download_dataset --base_dir download --csv_dir download --labels Cat Dog --format darknet --limit 500`