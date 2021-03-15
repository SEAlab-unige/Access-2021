# affordance-2021
This repository contains a demo of the proposed affordance network. The pretrained network is provided for easy usage. The network was trained using foregrounds of the object. Accordingly, input images are expected to contain the object in the foreground.

## Table of contents
* [Python project](#python-project)
* [References](#references)

## Python project
### Requirements
The requirements to run the python code are the following:
* Python 3.7 (64-bit)
* Tensorflow 2.X
* OpenCV

### Description
There are 3 folders:
* `models`: holds the main model for affordance detection (`MobileNetV1_UNET`) in *TFLite* format.
* `images`: holds some images used during the inference phase.
* `script`: holds two python files. `data_loader.py` consists of methods to perform image processing operations. `affordance_inference_tflite.py` performs the affordance prediction.

## References
Some (hopefully) useful links:
* [TFLite](https://www.tensorflow.org/lite)
* [TensorFlow Lite for Android](https://www.youtube.com/watch?v=JnhW5tQ_7Vo&feature=emb_logo)
