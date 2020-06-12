# Object detection tutorial with TensorFlow Object Detection API

Here I'll store notebooks and model files for the tutorial. 

The objective of this tutorial is to show how to train an object detection model on a custom dataset and convert it to `tflite` format. At the end of the tutorial, the model is compiled for inference on Coral Edge TPU devices.

The dataset I'll use is the Medical Masks Dataset available on [Kaggle](https://www.kaggle.com/vtech6/medical-masks-dataset).

## Structure of the tutorial

Part 1:
  - installing TensorFlow Object Detection API 
  - gathering and labelling data
  - converting the dataset to TFRecord format
  
Part 2:
  - configuring the training process
  - training and evaluating the model
  - run inference for frozen graph
  
Part 3:
  - exporting the model to TFLite and compiling it for Edge TPU
  - run inference for TFLite and compiled for Edge TPU format

