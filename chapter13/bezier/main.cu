#include <cuda.h>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

#define MAX_TESS_POINTS 32

// Contains all the parameters required to tessellate a Bezier line
struct BezierLine
{
	float2 CP[3];                      // control points
	float2 vertexPos[MAX_TESS_POINTS]; // vertex position array to tessellate into
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
void computeBezierLines(BezierLine* bLines, int nLines)
{
	int bidx = blockIdx.x;
	if (bidx < nLines) {
		// compute the curvature of the line
		float curvature = computeCurvature(bLines[bidx]);

		// compute the number of tessellation points from curvature (between 4 and 32)
		int nTessPoints = min(max((int)(curvature*16), 4), 32);
		bLines[bidx].nVertices = nTessPoints;

		// Loop through vertices to be tessellated
		for (int inc = 0; inc < nTessPoints; inc += blockDim.x) {
			int idx = inc + threadIdx.x; 
			if (idx < nTessPoints) {
				float u = (float)idx/(float)(nTessPoints - 1);
				float omu = 1.0f - u;
				float B3u[3];
				B3u[0] = omu*omu;
				B3u[1] = 2.0f*omu*u;
				B3u[2] = u*u;
				float2 position = {0, 0};
				for (int i = 0; i < 3; ++i)
					position = position + B3u[i]*bLines[bidx].CP[i];
				bLines[bidx].vertexPos[idx] = position;
			}
		}
	}
}

#define N_LINES 256
#define BLOCK_DIM 32

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
	cudaMalloc((void**)&bLines_d, N_LINES*sizeof(BezierLine));
	cudaMemcpy((void*)bLines_d, (void*)bLines_h.data(), N_LINES*sizeof(BezierLine), cudaMemcpyHostToDevice);

	// Call kernel
	computeBezierLines<<<N_LINES, BLOCK_DIM>>>(bLines_d, N_LINES);

	// Get results back on host side
	cudaMemcpy((void*)bLines_h.data(), (void*)bLines_d, N_LINES*sizeof(BezierLine), cudaMemcpyDeviceToHost);
	for (const auto& bLine : bLines_h) {
		cout << bLine;
	}

	// Release device memory
	cudaFree(bLines_d);

    cout << "Closing..." << endl;

    return 0;
}
