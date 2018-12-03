#include <cuda.h>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

__constant__ int BLUR_SIZE = 25;

/**
 * Convert the RGB image to gray scale (Kernel)
 */
__global__
void convertToGrayScaleKernel(unsigned char* Pbw, const unsigned char* Pin, int width, int height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if(col < width && row < height) {
    int grey_loc = row*width + col;
    int rgb_loc = 3*grey_loc;

    unsigned char r = Pin[rgb_loc  ];
    unsigned char g = Pin[rgb_loc+1];
    unsigned char b = Pin[rgb_loc+2];
    Pbw[grey_loc] = 0.21f*r + 0.71f*g + 0.07f*b;
  }
}

/**
 * Blur the BW image (Kernel)
 */
__global__
void blurKernel(unsigned char* Pout, const unsigned char* Pbw, int width, int height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if(col < width && row < height) {
    int loc = row*width + col;

    int pixval = 0;
    int pixels = 0;
    for(int i = -BLUR_SIZE; i < BLUR_SIZE+1; ++i) {
      for(int j = -BLUR_SIZE; j < BLUR_SIZE+1; ++j) {
        int blurRow = row + i;
        int blurCol = col + j;
        if(blurRow > -1 && blurRow < height && blurCol > -1 && blurCol < width) {
          pixval += Pbw[blurRow*width + blurCol];
          pixels++;
        }
      }
    }
    Pout[loc] = (unsigned char)(pixval/pixels);
  }
}

/**
 * Convert the RGB image to gray scale and blur
 */
unsigned char* convertToGrayScaleAndBlur(const unsigned char* h_Pin, int width, int height)
{
  // Allocate memory on host for out image
  int size = width*height;
  unsigned char* h_Pout = new unsigned char[size];

  // Allocate memory on device for in and bw images
  unsigned char* d_Pin;
  cudaMalloc((void**)&d_Pin, size*3);
  cudaMemcpy((void*)d_Pin, (void*)h_Pin, size*3, cudaMemcpyHostToDevice);
  unsigned char* d_Pbw;
  cudaMalloc((void**)&d_Pbw, size);

  // Run kernel BW
  dim3 dimGrid(ceil(width/16.0f),ceil(height/16.0f),1);
  dim3 dimBlock(16,16,1);
  cout << "Launching a (" << dimGrid.x << " x " << dimGrid.y << " x " << dimGrid.z << ") grid." << endl;
  cout << "Total number of threads: " << dimGrid.x*dimGrid.y*dimGrid.z*16*16 << endl;
  cout << "Number of pixels: " << width*height << endl;
  convertToGrayScaleKernel<<<dimGrid,dimBlock>>>(d_Pbw, d_Pin, width, height);

  // Allocated memory for blurred image
  unsigned char* d_Pout;
  cudaMalloc((void**)&d_Pout, size);

  // Run Kernel blur
  blurKernel<<<dimGrid,dimBlock>>>(d_Pout, d_Pbw, width, height);

  // Get the results back on host size
  cudaMemcpy((void*)h_Pout, (void*)d_Pout, size, cudaMemcpyDeviceToHost);

  cudaFree(d_Pin);
  cudaFree(d_Pbw);
  cudaFree(d_Pout);

  return h_Pout;
}

int main(int argc, char* argv[])
{
  // Read the image
  int width, height, channels;
  unsigned char* imageIn = stbi_load(argv[1], &width, &height, &channels, 0); 
  cout << "Imported image " << argv[1] << " (" << width << " x " << height << ") with " << channels << " channels" << endl;

  // convert the image to gray scale and blur
  unsigned char* imageOut = convertToGrayScaleAndBlur(imageIn, width, height);

  // display the final image
  stbi_write_bmp("image_out.bmp", width, height, 1, (void*)imageOut);

  // finalize images
  stbi_image_free(imageIn);
  delete [] imageOut;
  cout << "Closing..." << endl;

  return 0;
}
