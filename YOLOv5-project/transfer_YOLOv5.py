import glob
import os
import numpy as np
import shutil
from PIL import Image

#------------------------------------------------------------
# Create a folder structure for YOLOv5 training
#------------------------------------------------------------
if not os.path.exists('data'):
    for folder in ['images', 'labels']:
        for split in ['train', 'val', 'test']:
            os.makedirs(f'data/{folder}/{split}')

#------------------------------------------------------------
# Get filenames from the folder
#------------------------------------------------------------
def get_filenames(folder):
    filenames = set()
    for path in glob.glob(os.path.join(folder, '*.jpg')):
        filename = os.path.split(path)[-1]    # Extract the filename        
        filenames.add(filename)
    return filenames

dog_images = get_filenames('download/dog/images')
cat_images = get_filenames('download/cat/images')


# Clear Duplicates in Dog Dataset
duplicates = dog_images & cat_images
print("Duplicates: \n", duplicates)
dog_images -= duplicates



#---------- Convert the filename sets into Numpy ------------
dog_images = np.array(list(dog_images))
cat_images = np.array(list(cat_images))


#--------- Use the same random seed for reproducability -----
np.random.seed(42)
np.random.shuffle(dog_images)
np.random.shuffle(cat_images)



#---------- Split data into train, val, and test -----------
def split_dataset(animal, image_names, train_size, val_size):
    for i, image_name in enumerate(image_names):
        
        label_name = image_name.replace('.jpg', '.txt')
        
        if i < train_size:
            split = 'train'
        elif i < train_size + val_size:
            split = 'val'
        else:
            split = 'test'

        source_image_path = f'download/{animal}/images/{image_name}'
        source_label_path = f'download/{animal}/darknet/{label_name}'

        target_image_folder = f'data/images/{split}'
        target_label_folder = f'data/labels/{split}'
        
        shutil.copy(source_image_path, target_image_folder)
        shutil.copy(source_label_path, target_label_folder)


# Cat data and Dog data(reduce the number by 1 for each set due to three duplicates)
split_dataset('cat', cat_images, train_size=400, val_size=50)
split_dataset('dog', dog_images, train_size=399, val_size=49)






from PIL import Image, ImageDraw

def show_bbox(image_path):
    # convert image path to label path
    label_path = image_path.replace('/images/', '/labels/')
    label_path = label_path.replace('.jpg', '.txt')

    # Open the image and create ImageDraw object for drawing
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    with open(label_path, 'r') as f:
        for line in f.readlines():
            # Split the line into five values
            label, x, y, w, h = line.split(' ')

            # Convert string into float
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)

            # Convert center position, width, height into
            # top-left and bottom-right coordinates
            W, H = image.size
            x1 = (x - w/2) * W
            y1 = (y - h/2) * H
            x2 = (x + w/2) * W
            y2 = (y + h/2) * H

            # Draw the bounding box with red lines
            draw.rectangle((x1, y1, x2, y2),
                           outline=(255, 0, 0), # Red in RGB
                           width=5)             # Line width
    image.show()
    

show_bbox('data/images/train/0a0df46ca3f886c9.jpg')