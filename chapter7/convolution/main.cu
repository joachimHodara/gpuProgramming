#include <cuda.h>
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;

__global__
void convolution(const float* N, const float* M, float* P, int mask_width, int width)
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
//    vector<float> h_N(size);
//	std::iota(h_N.begin(), h_N.end(), 1.0f);

    // Copy vector on device side
    float* d_N;
    cudaMalloc((void**)&d_N, size*sizeof(float));
    cudaMemcpy((void*)d_N, (void*)h_N.data(), size*sizeof(float), cudaMemcpyHostToDevice);

	// Create mask and send to devide
	vector<float> h_M = { 1.0f, 1.0f, 2.0f, 1.0f, 1.0f };
	int mask_width = h_M.size();
    float* d_M;
    cudaMalloc((void**)&d_M, mask_width*sizeof(float));
    cudaMemcpy((void*)d_M, (void*)h_M.data(), mask_width*sizeof(float), cudaMemcpyHostToDevice);

	// Allocate space for solution on device
    float* d_P;
    cudaMalloc((void**)&d_P, size*sizeof(float));

    // call Kernel
	int blockDim = 4;
	int gridDim = ceil(size/(float)blockDim);
	convolution<<<gridDim, blockDim>>>(d_N, d_M, d_P, mask_width, size);

    // Recover vector from device to host
	vector<float> h_P(size);
    cudaMemcpy((void*)h_P.data(), (void*)d_P, size*sizeof(float), cudaMemcpyDeviceToHost);

    // Check results
	for (auto elt : h_P)
		cout << elt << " ";
	cout << endl;

    // Finalize storage
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    cout << "Closing..." << endl;

    return 0;
}
