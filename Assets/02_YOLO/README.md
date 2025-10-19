---
license: gpl-3.0
library_name: unity-sentis
pipeline_tag: object-detection
tags:
  - unity-inference-engine
---
# YOLOv8, YOLOv9, YOLO11, YOLO12 in Unity 6 using Inference Engine

[YOLO](https://docs.ultralytics.com/models/) is a real-time multi-object recognition model.
Small and Nano model sizes are included for YOLO version 8 and above (except version 10 which uses NMS-free approach).

## How to Use

* Create a new scene in Unity 6;
* Install `com.unity.ai.inference` from the package manager;
* Add the `RunYOLO.cs` script to the Main Camera;
* Drag an appropriate `.onnx` file from the `models` folder into the `Model Asset` field;
* Drag the `classes.txt` file into the `Classes Asset` field;
* Create a `GameObject > UI > Raw Image` object in the scene, set its width and height to 640, and link it as the `Display Image` field;
* Drag the `Border Texture.png` file into the `Border Texture` field;
* Select an appropriate font in the `Font` field;
* Put a video file in the `Assets/StreamingAssets` folder and set the `Video Filename` field to the filename of the video.

## Preview
Enter play mode. If working correctly you should see something like this:

![preview](preview.jpg)

## Inference Engine
Inference Engine is a neural network inference library for Unity. Find out more [here](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest).