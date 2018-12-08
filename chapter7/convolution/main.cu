#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;

#define TILE_SIZE 64
#define MAX_MASK_WIDTH 5

__constant__ float c_M[MAX_MASK_WIDTH];

__global__
void convolution1(const float* N, const float* M, float* P, int mask_width, int width)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    float Pvalue = 0.0f;
    int N_start_point = i - (mask_width/2);
    for (int j = 0; j < mask_width; ++j) {
        if (N_start_point + j >= 0 && N_start_point + j < width) {
            Pvalue += N[N_start_point + j]*M[j];
        }
    }
    P[i] = Pvalue;
}

__global__
void convolution2(const float* N, float* P, int mask_width, int width)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    float Pvalue = 0.0f;
    int N_start_point = i - (mask_width / 2);
    for (int j = 0; j < mask_width; ++j) {
        if (N_start_point + j >= 0 && N_start_point + j < width) {
            Pvalue += N[N_start_point + j] * c_M[j];
        }
    }
    P[i] = Pvalue;
}

__global__
void convolution3(const float* N, float* P, int mask_width, int width)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float Nds[TILE_SIZE + MAX_MASK_WIDTH - 1];

    // load N from global memory into shared memory
    int n = mask_width/2;

    if (threadIdx.x >= blockDim.x - n) {
        int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
        Nds[threadIdx.x - (blockDim.x - n)] = (halo_index_left < 0) ? 0 : N[halo_index_left];
    }

    Nds[n + threadIdx.x] = N[i];

    if (threadIdx.x < n) {
        int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
        Nds[n + blockDim.x + threadIdx.x] = (halo_index_right >= width) ? 0 : N[halo_index_right];
}

    __syncthreads();

    float Pvalue = 0.0f;
    for (int j = 0; j < mask_width; ++j) {
        Pvalue += Nds[threadIdx.x + j]*c_M[j];
    }
    P[i] = Pvalue;
}

int main(int argc, char* argv[])
{
    // Query GPU properties
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    cout << "---------------------------------------------" << endl;
    cout << "               GPU PROPERTIES                " << endl;
    cout << "---------------------------------------------" << endl;
    cout << "Device Name: " << dev_prop.name << endl;
    cout << "Memory Clock Rate: " << dev_prop.memoryClockRate/1.0e6 <<  " GHz" << endl;
    cout << "Memory Bandwidth: " << 2.0*dev_prop.memoryClockRate*(dev_prop.memoryBusWidth/8)/1.0e6 <<  " GB/s" << endl;
    cout << "Number of SM: " << dev_prop.multiProcessorCount << endl;
    cout << "Max Threads per SM: " << dev_prop.maxThreadsPerMultiProcessor << endl;
    cout << "Registers per Block: " << dev_prop.regsPerBlock << endl;
    cout << "Shared Memory per Block: " << dev_prop.sharedMemPerBlock << " B" << endl;
    cout << "Total Global Memory per Block: " << dev_prop.totalGlobalMem/1.0e9 << " GB" << endl;
    cout << endl;

    int size = atoi(argv[1]);

    // creating vector on host side
    vector<float> h_N(size, 1.0f);
    std::iota(h_N.begin(), h_N.end(), 0.0f);

    // Copy vector on device side
    float* d_N;
    cudaMalloc((void**)&d_N, size*sizeof(float));
    cudaMemcpy((void*)d_N, (void*)h_N.data(), size*sizeof(float), cudaMemcpyHostToDevice);

    // Create mask and send to devide
    vector<float> h_M = { 1.0f, 1.0f, 2.0f, 1.0f, 1.0f };
    int mask_width = h_M.size();
    assert(mask_width < MAX_MASK_WIDTH);
    cudaMemcpyToSymbol(c_M, (void*)h_M.data(), mask_width*sizeof(float));

    // Allocate space for solution on device
    float* d_P;
    cudaMalloc((void**)&d_P, size*sizeof(float));

    // call Kernel
    int blockDim = TILE_SIZE;
    int gridDim = ceil(size/(float)blockDim);
    int version = atoi(argv[2]);
    if(version == 1)
        convolution2<<<gridDim, blockDim>>>(d_N, d_P, mask_width, size);
    else if(version == 2)
        convolution3<<<gridDim, blockDim>>>(d_N, d_P, mask_width, size);
    else
        cout << "Wrong inputs!" << endl;

    // Recover vector from device to host
    vector<float> h_P(size);
    cudaMemcpy((void*)h_P.data(), (void*)d_P, size*sizeof(float), cudaMemcpyDeviceToHost);

    // Finalize storage
    cudaFree(d_N);
    cudaFree(d_P);

    cout << "Closing..." << endl;

    return 0;
}
