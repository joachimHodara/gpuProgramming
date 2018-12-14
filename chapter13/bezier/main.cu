#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

#define N_LINES 256
#define MAX_TESS_POINTS 32
#define BLOCK_DIM 32

// Contains all the parameters required to tessellate a Bezier line
struct BezierLine
{
	float2 CP[3];                      // control points
	float2* vertexPos;                 // vertex position array to tessellate into
	int nVertices;                     // number of tessellated vertices
};

ostream& operator<<(ostream& out, const BezierLine& bLine)
{
	out << "Bezier line: " << endl;
	out << bLine.CP[0].x << " " << bLine.CP[0].y << endl;
	out << bLine.CP[1].x << " " << bLine.CP[1].y << endl;
	out << bLine.CP[2].x << " " << bLine.CP[2].y << endl;
	out << "Tesselation: " << bLine.nVertices << " vertices" << endl;
	for (int i = 0; i < bLine.nVertices; ++i)
		out << bLine.vertexPos[i].x << " " << bLine.vertexPos[i].y << endl;
	return out;
}

__forceinline__ __device__
float2 operator+(float2 a, float2 b)
{
	a.x += b.x;
	a.y += b.y;
	return a;
}

__forceinline__ __device__
float2 operator-(float2 a, float2 b)
{
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

__forceinline__ __device__
float2 operator*(float a, float2 b)
{
	b.x *= a;
	b.y *= a;
	return b;
}

__forceinline__ __device__
float length(float2 a)
{
	return sqrtf(a.x*a.x + a.y*a.y);
}

__device__
float computeCurvature(const BezierLine& bLine)
{
	return length(bLine.CP[1] - 0.5f*(bLine.CP[0] + bLine.CP[2])) / length(bLine.CP[2] - bLine.CP[0]);
}

__global__
void computeBezierLines_child(int lidx, BezierLine* bLines, int nTessPoints)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < nTessPoints) {
		float u = (float)idx / (float)(nTessPoints - 1);
		float omu = 1.0f - u;
		float B3u[3];
		B3u[0] = omu*omu;
		B3u[1] = 2.0f*omu*u;
		B3u[2] = u*u;
		float2 position = { 0, 0 };
		for (int i = 0; i < 3; ++i)
			position = position + B3u[i]*bLines[lidx].CP[i];
		bLines[lidx].vertexPos[idx] = position;
	}
}

__global__
void computeBezierLines_parent(BezierLine* bLines, int nLines)
{
	int lidx = blockIdx.x*blockDim.x + threadIdx.x;
	if (lidx < nLines) {
		// compute the curvature of the line
		float curvature = computeCurvature(bLines[lidx]);

		// compute the number of tessellation points from curvature (between 4 and 32)
		bLines[lidx].nVertices = min(max((int)(curvature * 16), 4), MAX_TESS_POINTS);
		cudaMalloc((void**)&bLines[lidx].vertexPos, bLines[lidx].nVertices*sizeof(float2));

		// Call child kernel to do the actual math
		cudaStream_t stream;
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		computeBezierLines_child<<<ceil(bLines[lidx].nVertices/32.0f),32, 0, stream>>>(lidx, bLines, bLines[lidx].nVertices);
		cudaStreamDestroy(stream);
	}
}

__global__
void freeVertexMem(BezierLine* bLines, int nLines)
{
	int lidx = threadIdx.x + blockIdx.x*blockDim.x;
	if (lidx < nLines)
		cudaFree(bLines[lidx].vertexPos);
}

// Initialize Bezier lines randomly
void initializeBLines(vector<BezierLine>& bLines)
{
	default_random_engine generator;
	uniform_real_distribution<float> distrib(0, 1.0f);

	float2 last = { 0, 0 };
	for (auto& bLine : bLines) {
		bLine.CP[0] = last;
		for (int j = 1; j < 3; ++j) {
			bLine.CP[j].x = distrib(generator);
			bLine.CP[j].y = distrib(generator);
		}
		last = bLine.CP[2];
		bLine.nVertices = 0;
	}
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

	// Allocate Bezier lines on host
	vector<BezierLine> bLines_h(N_LINES);
	initializeBLines(bLines_h);

	// Allocate and copy BLines to device
	BezierLine* bLines_d;
	cudaMalloc((void**)&bLines_d, N_LINES * sizeof(BezierLine));
	cudaMemcpy((void*)bLines_d, (void*)bLines_h.data(), N_LINES * sizeof(BezierLine), cudaMemcpyHostToDevice);

	// Call kernel
	cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, N_LINES);
	computeBezierLines_parent<<<(float)N_LINES/(float)BLOCK_DIM, BLOCK_DIM>>>(bLines_d, N_LINES);

	// Release device memory
	freeVertexMem<<<(float)N_LINES/(float)BLOCK_DIM, BLOCK_DIM>>>(bLines_d, N_LINES);
	cudaFree(bLines_d);

    cout << "Closing..." << endl;

    return 0;
}
