#!/bin/bash

# Compile first
echo "Compiling..."
make

# Check if an input image exists
INPUT_IMG="data/input/test.jpg"

if [ ! -f "$INPUT_IMG" ]; then
    echo "Error: $INPUT_IMG not found!"
    echo "Please download a test image (e.g., Lena or Mandrill) and place it in data/input/test.jpg"
    exit 1
fi

# Run Blur
echo "--- Running Mode 1: Gaussian Blur ---"
./build/image_proc "$INPUT_IMG" "data/output/blurred.jpg" 1

# Run Edge Detection
echo "--- Running Mode 2: Sobel Edge Detection ---"
./build/image_proc "$INPUT_IMG" "data/output/edges.jpg" 2

echo "Done! Check data/output/ for results."