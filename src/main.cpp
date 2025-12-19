#include <iostream>
#include <string>
#include <chrono> // For timing
#include <cuda_runtime.h> // For timing
#include "kernels.h"

// STB Image definitions
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Prototype for the wrapper in kernels.cu
void launchProcessing(unsigned char* d_input, unsigned char* d_output, int width, int height, int mode);

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image> <mode>" << std::endl;
        std::cerr << "Modes: 1 = Gaussian Blur, 2 = Sobel Edge Detection" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    int mode = std::stoi(argv[3]);

    // 1. Load Image (Force Grayscale for simplicity: 1 channel)
    int width, height, channels;
    unsigned char* h_input = stbi_load(inputFile.c_str(), &width, &height, &channels, 1);
    
    if (!h_input) {
        std::cerr << "Error loading image: " << inputFile << std::endl;
        return 1;
    }

    size_t imgSize = width * height * sizeof(unsigned char);
    std::cout << "Image Loaded: " << width << "x" << height << " (Grayscale)" << std::endl;

    // 2. Allocate GPU Memory
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, imgSize);
    cudaMalloc((void**)&d_output, imgSize);

    // 3. Copy Data to GPU
    cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice);

    // 4. Run Kernel & Measure Time
    auto start = std::chrono::high_resolution_clock::now();
    
    launchProcessing(d_input, d_output, width, height, mode);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "GPU Processing Time: " << duration.count() << " ms" << std::endl;

    // 5. Copy Result Back to Host
    // Reuse h_input buffer for output to save memory, or allocate new if needed
    unsigned char* h_output = (unsigned char*)malloc(imgSize);
    cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost);

    // 6. Save Image
    stbi_write_jpg(outputFile.c_str(), width, height, 1, h_output, 100);
    std::cout << "Saved output to: " << outputFile << std::endl;

    // 7. Cleanup
    stbi_image_free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}