# Azure Kinect Camera Test Cases

This repository contains a set of Python scripts designed to test and experiment with different functionalities of the Azure Kinect camera. The primary focus of these test cases is detecting, tracking, and processing data related to the depth, arm, and facial recognition using various models, including YOLO and MobileNet.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [License](#license)

## Overview
This project is a collection of test scripts that utilize the Azure Kinect camera for different purposes such as arm detection, face tracking, depth data analysis, and object detection using machine learning models. The tests make use of libraries like MediaPipe, OpenCV, and YOLO.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have the Azure Kinect SDK installed and properly configured on your system.

## Usage

Each script has its own purpose, which can be run independently for testing various features. Below are details about the most important files and their functionalities.

For example:
```bash
python MediapipeArmtrial.py
```

You can experiment with the provided scripts to capture depth data, detect arms, or test different YOLO-based object detection tasks.

## File Descriptions

- **MediapipeArmtrial.py**: Script using MediaPipe for detecting and tracking arms in the camera's field of view.
- **MobileNet_SSDtrial.py**: A trial script that uses the MobileNet SSD model for object detection.
- **YOLOtrial.py**: Implements YOLO for real-time object detection using the Azure Kinect camera feed.
- **anglebyCoordients.py**: Computes angles between coordinates detected in the camera's field of view, potentially used for pose estimation or arm angle calculations.
- **benchmark.py**: Script for testing and benchmarking the performance of the Azure Kinect in various scenarios.
- **dummyArmtrial.py**: Experimental script for dummy arm detection or testing.
- **facetrial.py**: Implements facial recognition or tracking features using the Azure Kinect.
- **helpers.py**: A utility script containing helper functions for common tasks such as image or data processing.
- **viewer.py**: Displays camera feeds, and potentially outputs information such as object tracking or detection in real-time.
- **viewer_depth.py**: Focuses on capturing and displaying depth information from the Azure Kinect camera.
- **playback.py**: Provides functionality to playback recorded footage or streams from the Azure Kinect for testing purposes.
- **yolov3.cfg / yolov3.weights**: Configuration and weight files for running YOLOv3 object detection. These are essential for object detection tasks using YOLO.
