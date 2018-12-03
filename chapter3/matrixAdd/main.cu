#include <cuda.h>
#include <iostream>
using namespace std;

/**
 * C = A + B (one element per thread)
 */
__global__
void addMatricesElt(float* C, const float* A, const float* B, int dim)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if(col < dim && row < dim) {
    int i = row*dim + col;
	C[i] = A[i] + B[i];
  }
}

/**
* C = A + B (one row per thread)
*/
__global__
void addMatricesRow(float* C, const float* A, const float* B, int dim)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < dim) {
		int rowdim = row*dim;
		for (int col = 0; col < dim; ++col) {
                        int i = rowdim + col;
			C[i] = A[i] + B[i];
		}
	}
}

/**
* C = A + B (one column per thread)
*/
__global__
void addMatricesCol(float* C, const float* A, const float* B, int dim)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < dim) {
		for (int row = 0; row < dim; ++row) {
			int i = row*dim + col;
			C[i] = A[i] + B[i];
		}
	}
}

int main(int argc, char* argv[])
{
	int dim = atoi(argv[1]);
	int size = dim*dim;

	// creating matrices on host side
	float* h_A = new float[size];
	float* h_B = new float[size];
	for (int i = 0; i < size; ++i) {
		h_A[i] = 1.0f;
		h_B[i] = 2.0f;
	}

	// Copy matrices on device side
	float* d_A;
	cudaMalloc((void**)&d_A, size*sizeof(float));
	cudaMemcpy((void*)d_A, (void*)h_A, size*sizeof(float), cudaMemcpyHostToDevice);
	float* d_B;
	cudaMalloc((void**)&d_B, size*sizeof(float));
	cudaMemcpy((void*)d_B, (void*)h_B, size*sizeof(float), cudaMemcpyHostToDevice);

	// Allocate C matrix on device
	float* d_C;
	cudaMalloc((void**)&d_C, size*sizeof(float));

	// call Kernel
	int type = atoi(argv[2]);
	if (type == 1) { // one element per thread
		dim3 dimGrid(ceil(dim/16.0f), ceil(dim/16.0f), 1);
		dim3 dimBlock(16, 16, 1);
		addMatricesElt<<<dimGrid, dimBlock>>> (d_C, d_A, d_B, dim);
	}
	else if (type == 2) { // one row per thread
		dim3 dimGrid(1, ceil(dim/256.0f), 1);
		dim3 dimBlock(1, 256, 1);
		addMatricesRow<<<dimGrid, dimBlock>>> (d_C, d_A, d_B, dim);
	}
	else if (type == 3) { // one column per thread
		dim3 dimGrid(ceil(dim/256.0f), 1, 1);
		dim3 dimBlock(256, 1, 1);
		addMatricesCol<<<dimGrid, dimBlock>>> (d_C, d_A, d_B, dim);
	}
	else
		cout << "invalid argument!" << endl;

	// Recover C matrix from device to host
	float* h_C = new float[size];
	cudaMemcpy((void*)h_C, (void*)d_C, size*sizeof(float), cudaMemcpyDeviceToHost);

	// Check results
	for (int i = 0; i < size; ++i) {
		if (fabs(h_C[i] - 3.0f) > 0.0001f) {
			cout << "ERROR: something is not right." << endl;
			break;
		}
	}

	// Finalize storage
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	delete [] h_A;
	delete [] h_B;
	delete [] h_C;

	cout << "Closing..." << endl;

	return 0;
}
