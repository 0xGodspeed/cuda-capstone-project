#include <iostream>
#include "kernels.h"

// Define STB libraries here (only once in the entire project)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char* argv[]) {
    std::cout << "--- CUDA Capstone Project Starting ---" << std::endl;

    // 1. Test GPU Connection
    std::cout << "Testing GPU..." << std::endl;
    launchTestKernel();

    // 2. Test Image Library
    std::cout << "STB Libraries loaded successfully." << std::endl;

    return 0;
}