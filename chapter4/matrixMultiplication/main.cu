#include <cuda.h>
#include <iostream>
using namespace std;

__constant__ int TILE_WIDTH = 10;

/**
 * C = A * B 
 */
__global__
void matMul(float* C, const float* A, const float* B, int dim)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if(col < dim && row < dim) {
	  float prod = 0.0f;
	  for(int i = 0; i < dim; i++)
		  prod += A[row*dim+i]*B[i*dim+col];
	  C[row*dim+col] = prod;
  }
}

/**
 * C = A * B (tiled)
 */
__global__
void matMulTiled(float* C, const float* A, const float* B, int dim)
{
	__shared__ float As[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;  int tx = threadIdx.x;
	int by = blockIdx.y;  int ty = threadIdx.y;

	int row = by*TILE_WIDTH + ty;
	int col = bx*TILE_WIDTH + tx;

	// Loop over the tiles required to compute the element
	float prod = 0.0f;
	for(int ph = 0; ph < ceil(dim/(float)TILE_WIDTH); ++ph) {
		
		// 1. Load the tiles into shared memory
		int pcol = ph*TILE_WIDTH + tx;
		if(row < dim && pcol < dim)
			As[ty][tx] = A[row*dim + pcol];
		int prow = ph*TILE_WIDTH + ty;
		if(prow < dim && col < dim)
			Bs[ty][tx] = B[prow*dim + col];
		__syncthreads();

		// 2. Dot product
		for(int i = 0; i < TILE_WIDTH; ++i)
			prod += As[ty][i]*Bs[i][tx];
		__syncthreads();

	}

	// 3. Write result
	if(row < dim && col < dim) C[row*dim+col] = prod;
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
		h_A[i] = 3.0f;
		h_B[i] = 0.0f;
	}
	for (int i = 0; i < size; i+=dim+1)
		h_B[i] = 1.0f;

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
	if (type == 1) { // "regular" matrix multiplication
		dim3 dimGrid(ceil(dim/16.0f), ceil(dim/16.0f), 1);
		dim3 dimBlock(16, 16, 1);
		matMul<<<dimGrid, dimBlock>>> (d_C, d_A, d_B, dim);
	}
	else if (type == 2) { // "tiled" matrix multiplication
		dim3 dimGrid(ceil(dim/16.0f), ceil(dim/16.0f), 1);
		dim3 dimBlock(16, 16, 1);
		matMulTiled<<<dimGrid, dimBlock>>> (d_C, d_A, d_B, dim);
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
