# target-detector-yolov5

YouTube Video:

[![Youtube Video](https://img.youtube.com/vi/AaqFqx3wzfE/0.jpg)](https://www.youtube.com/watch?v=AaqFqx3wzfE)


This script is a video processing tool that uses the YOLOv5 object detection model to detect specified objects in a video. It allows the user to define a polygonal region of interest (ROI) and detects if a person is inside it. If a person is detected, an alarm sound is played and photos of the detected image is saved. The script also displays the video feed with the detected objects and their labels. This tool can be used for security purposes. The script requires the installation of OpenCV, Pygame, and PyTorch libraries.

Since the test videos' sizes are too big, I couldn't upload them.
Do not forget to change the paths of the alarm sound, detected photos, test videos depending on how you store the script


The model used: yolov5s.pt
https://github.com/ultralytics/yolov5
