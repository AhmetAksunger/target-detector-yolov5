# target-detector-yolov5
An object detection script using yolov5 model. If any spesificed object gets into the spesificed area, an alarm triggers and script takes photos of the target.

YouTube Video:

[![Youtube Video](https://img.youtube.com/vi/AaqFqx3wzfE/0.jpg)](https://www.youtube.com/watch?v=AaqFqx3wzfE)


This script is a video processing tool that uses the YOLOv5 object detection model to detect specified objects in a video. It allows the user to define a polygonal region of interest (ROI) and detects if a person is inside it. If a person is detected, an alarm sound is played and photos of the detected image is saved. The script also displays the video feed with the detected objects and their labels. This tool can be used for security purposes. The script requires the installation of OpenCV, Pygame, and PyTorch libraries.

The model used: yolov5s.pt
https://github.com/ultralytics/yolov5
