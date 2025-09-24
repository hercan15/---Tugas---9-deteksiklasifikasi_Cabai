
Kematangan Cabai - v6 2025-07-06 4:19pm
==============================

This dataset was exported via roboflow.com on July 6, 2025 at 8:21 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 3504 images.
Cabai are annotated in YOLOv8 Oriented Object Detection format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Fill (with center crop))
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 3 versions of each source image:
* Randomly crop between 0 and 20 percent of the image
* Random brigthness adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 2.5 pixels
* Salt and pepper noise was applied to 0.1 percent of pixels


