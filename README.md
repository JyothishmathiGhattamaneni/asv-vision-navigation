# ASV Vision System for Shoreline Detection and Obstacle Identification

## Project Overview
This project simulates an Autonomous Surface Vehicle (ASV) using MATLAB
and implements live shoreline detection and potential obstacle identification
using Python and OpenCV.

## Features
- MATLAB-based ASV simulation
- Four camera views: front, rear, left, and right
- Live camera-frame exchange between MATLAB and Python
- Shoreline detection using image processing
- Potential obstacle identification
- Annotated camera feeds displayed in MATLAB

## Technologies Used
- MATLAB R2025b
- Simulink 3D Animation
- Python
- OpenCV
- NumPy

## Project Files
- `asv_four_camera_live_detection_10ms_5sec.m`:
  MATLAB simulation and camera-frame exchange.
- `live_water_boundary_detector_v4.py`:
  Python shoreline and obstacle detection.

## How to Run
1. Install Python and the required Python libraries.
2. Keep both files in the same project folder.
3. Start the Python detector in a terminal.
4. Run the MATLAB simulation.
5. Observe the four camera feeds and detection overlays.

## Limitations
The system provides experimental visual detections and has not been
validated for real-world autonomous navigation or safety-critical use.
