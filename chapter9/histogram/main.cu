#include <cuda.h>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

__global__
void histoKernel(int* data, int* count, int size)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if(tid < size)
        atomicAdd(&count[data[tid]/4], 1);
}

__global__
void histoKernelPrivatized(int* data, int* count, int size)
{
    __shared__ int count_s[8];

    int t = threadIdx.x;
    int tid = blockIdx.x*blockDim.x + t;

    if(t < 8) 
        count_s[t] = 0;
    __syncthreads();

    if(tid < size)
        atomicAdd(&count_s[data[tid]/4], 1);
    __syncthreads();

    if(t < 8)
        atomicAdd(&count[t], count_s[t]);
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

    default_random_engine generator;
    uniform_int_distribution<int> distrib(0,31);

    vector<int> h_data(size);
    for(auto& elt : h_data) elt = distrib(generator);

    int* d_data;
    cudaMalloc((void**)&d_data, size*sizeof(int));
    cudaMemcpy((void*)d_data, (void*)h_data.data(), size*sizeof(int), cudaMemcpyHostToDevice);

    vector<int> h_count(8,0);
    int* d_count;
    cudaMalloc((void**)&d_count, 8*sizeof(int));
    cudaMemcpy((void*)d_count, (void*)h_count.data(), 8*sizeof(int), cudaMemcpyHostToDevice);

    int type = atoi(argv[2]);
    int blockSize = 256;
    if(type == 1)
        histoKernel<<<ceil(size/(float)blockSize),blockSize>>>(d_data, d_count, size);
    else if (type == 2)
        histoKernelPrivatized<<<ceil(size/(float)blockSize),blockSize>>>(d_data, d_count, size);
    else
        cout << "wrong input!" << endl;

    cudaMemcpy((void*)h_count.data(), (void*)d_count, 8*sizeof(int), cudaMemcpyDeviceToHost);

    for(auto elt : h_count) cout << elt << " ";
    cout << endl;

    cudaFree(d_data);
    cudaFree(d_count);
    cout << "Closing..." << endl;

    return 0;
}
