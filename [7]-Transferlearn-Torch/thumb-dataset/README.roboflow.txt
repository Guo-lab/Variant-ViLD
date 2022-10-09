
thumb - v4 2022-09-01 2:35pm
==============================

This dataset was exported via roboflow.com on October 9, 2022 at 7:42 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 453 images.
Hand are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Random brigthness adjustment of between -45 and +45 percent
* Random exposure adjustment of between -30 and +30 percent


