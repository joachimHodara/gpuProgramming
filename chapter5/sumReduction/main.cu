#include <cuda.h>
#include <iostream>
#include <vector>
using namespace std;

__global__
void sumReduction(float* v, int size, int jump)
{
    // linear id
    unsigned int t = threadIdx.x;
	unsigned int t0 = blockIdx.x*blockDim.x;
    unsigned int k = jump*(t0 + t);

    // load vector into shared memory
    extern __shared__ float vs[]; 
    vs[t] = v[k];

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if(t % (2*stride) == 0)
            vs[t] += vs[t + stride];
    }

	if (t == 0)
		v[jump*t0] = vs[0];
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
    vector<float> vec(size, 1.0f);

    // Copy vector on device side
    float* d_vec;
    cudaMalloc((void**)&d_vec, size*sizeof(float));
    cudaMemcpy((void*)d_vec, (void*)vec.data(), size*sizeof(float), cudaMemcpyHostToDevice);

    // call Kernel
	int blockDim = 4;
	int jump = 1;
	int number_of_blocks = size;
	do {
		number_of_blocks = ceil(number_of_blocks/(float)blockDim);
		sumReduction<<<number_of_blocks, blockDim, blockDim*sizeof(float)>>>(d_vec, size, jump);
		jump *= 4;
	} while (number_of_blocks != 1);

    // Recover vector from device to host
    cudaMemcpy((void*)vec.data(), (void*)d_vec, size*sizeof(float), cudaMemcpyDeviceToHost);

    // Check results
    if (fabs(vec[0] - size) > 0.0001f)
        cout << "ERROR: something is not right." << endl;

    // Finalize storage
    cudaFree(d_vec);

    cout << "Closing..." << endl;

    return 0;
}
