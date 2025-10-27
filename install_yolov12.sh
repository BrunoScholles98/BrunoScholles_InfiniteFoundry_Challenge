#!/bin/bash
# Script to clone and install YOLOv12

# Stop the script if any command fails
set -e

# Clone the repository
git clone https://github.com/sunsmarterjie/yolov12.git

# Enter the repository directory
cd yolov12/

# Install the package in editable mode
pip install -e .