import torch

# Download YOLOv5 from PyTorch Hub
# Stored in ~/.cache/torch/hub/ultralytics_yolov5_master/
# and torch.hub.load will fetch the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Sample Image URL
BASE_URL = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
image_url = BASE_URL + 'zidane.jpg'

local_image_path = "/Users/gsq/Desktop/Alex_Vild/Variant-ViLD/[8]-YOLOv5-project/raw_image/R-C.jpeg"

imgs = [image_url, local_image_path]

results = model(imgs)
results.show()
#//results.save()

results.print()
print(results.xyxy[0])
print(model.names)