#include "kernels.h"
#include <math.h>
#include <iostream>

// --- Helper: Gaussian Blur Kernel (3x3) ---
__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int pixVal = 0;
        
        // 3x3 Gaussian Kernel (Approximate)
        // 1 2 1
        // 2 4 2
        // 1 2 1
        int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
        int kernelSum = 16;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int offset = ((y + j) * width + (x + i));
                pixVal += input[offset] * kernel[i + 1][j + 1];
            }
        }
        output[y * width + x] = (unsigned char)(pixVal / kernelSum);
    }
}

// --- Helper: Sobel Edge Detection Kernel ---
__global__ void sobelKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

        float gradX = 0.0f;
        float gradY = 0.0f;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int offset = ((y + j) * width + (x + i));
                unsigned char val = input[offset];
                gradX += val * Gx[i + 1][j + 1];
                gradY += val * Gy[i + 1][j + 1];
            }
        }

        // Calculate Magnitude
        int magnitude = (int)sqrtf(gradX * gradX + gradY * gradY);
        
        // Clamp to 0-255
        if (magnitude > 255) magnitude = 255;
        if (magnitude < 0) magnitude = 0;

        output[y * width + x] = (unsigned char)magnitude;
    }
}

// --- Wrapper Function ---
void launchProcessing(unsigned char* d_input, unsigned char* d_output, int width, int height, int mode) {
    // Define Block Size (Standard 16x16 or 32x32)
    dim3 blockSize(16, 16);
    
    // Calculate Grid Size to cover the whole image
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);

    if (mode == 1) {
        printf("Launching Gaussian Blur Kernel...\n");
        gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    } else if (mode == 2) {
        printf("Launching Sobel Edge Detection Kernel...\n");
        sobelKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
    }
}