# High-Performance GPU Image Convolution Engine

## Project Overview
This project is a high-performance image processing application built using **CUDA C++**. It leverages the massive parallelism of NVIDIA GPUs to perform computationally intensive convolution operations—specifically Gaussian Blur and Sobel Edge Detection—on high-resolution images. 

Unlike standard CPU-based image processing, which processes pixels sequentially, this engine assigns individual threads to pixels, allowing for near-instantaneous transformation of 4K and 8K imagery. This project demonstrates core GPU concepts including:
* Global Memory Management (`cudaMalloc`, `cudaMemcpy`)
* 2D Kernel Mapping (Grids and Blocks)
* Convolution Algorithms
* Unified Host/Device Logic

## Features
* **Gaussian Blur:** Applies a 3x3 convolution kernel to reduce image noise and detail.
* **Sobel Edge Detection:** Calculates gradient magnitude to highlight edges (essential for Computer Vision tasks).
* **CLI Interface:** Fully command-line driven for automation and batch processing.
* **Format Support:** Handles standard formats (JPG, PNG, BMP) via the `stb_image` library.

## Prerequisites
* **OS:** Linux (Ubuntu 18.04/20.04 recommended)
* **Hardware:** NVIDIA GPU (Compute Capability 6.0+)
* **Software:** * CUDA Toolkit (10.0+)
    * GCC Compiler
    * Make

## Compilation Instructions
This project uses a `Makefile` for easy compilation.

```bash
# Clean previous builds
make clean

# Compile the project
make all