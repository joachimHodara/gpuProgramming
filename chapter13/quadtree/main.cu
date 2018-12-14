#include <cuda.h>
#include <iostream>
using namespace std;

// Add two float2
__host__ __device__ __forceinline__ float2 operator+(float2 a, float2 b) 
{ 
	a.x += b.x;
	a.y += b.y;
	return a;
}

// Multiply a float2 and a float
__host__ __device__ __forceinline__ float2 operator*(float a, float2 b)
{
	b.x *= a;
	b.y *= a;
	return b;
}

// a structure of 2d points
class Points
{
public:

	// constructor
	__host__ __device__ Points(float* x = nullptr, float* y = nullptr) : m_x(x), m_y(y) {}

	// get a point
	__host__ __device__ __forceinline__ float2 get_point(int idx) const { return make_float2(m_x[idx], m_y[idx]); }

	// set a point
	__host__ __device__ __forceinline__ void set_point(int idx, const float2& p) { m_x[idx] = p.x; m_y[idx] = p.y; }

	// set the points pointers
	__host__ __device__ __forceinline__ void set(float* x, float* y) { m_x = x; m_y = y; }

private:

	// coordinates
	float* m_x;
	float* m_y;
};

// a 2D bounding box
class Bounding_box
{
public:

	// constructor
	__host__ __device__ Bounding_box() : m_p_min(make_float2(0, 0)), m_p_max(make_float2(1.0f, 1.0f)) {}

	// compute the center of the box
	__host__ __device__ float2 computeCenter() const { return 0.5f*(m_p_min + m_p_max); }

	// Get the corner points
	__host__ __device__ __forceinline__ const float2& get_max() const { return m_p_max; }
	__host__ __device__ __forceinline__ const float2& get_min() const { return m_p_min; }

	// Does the box contains the input point?
	__host__ __device__ bool contains(const float2& p) const 
	{ 
		return (p.x >= m_p_min.x) && (p.x < m_p_max.x)
			&& (p.y >= m_p_min.y) && (p.y < m_p_max.y);
	}

	// Define the boundinx box
	__host__ __device__ void set(float min_x, float min_y, float max_x, float max_y)
	{
		m_p_min.x = min_x;
		m_p_min.y = min_y;
		m_p_max.x = max_x;
		m_p_max.y = max_y;
	}

private:

	// corners
	float2 m_p_min;
	float2 m_p_max;
};

// A node in the quadtree
class Quadtree_node
{
public:

	// constructor
	__host__ __device__ Quadtree_node() : m_id(0), m_begin(0), m_end(0) {}

	// get the node id
	__host__ __device__ int id() const { return m_id; }

	// set the node id
	__host__ __device__ void set_id(int new_id) { m_id = new_id; }

	// get the bounding box
	__host__ __device__ __forceinline__ const Bounding_box& bounding_box() const { return m_bounding_box; }

	// set the bounding box
	__host__ __device__ __forceinline__ void set_bounding_box(float min_x, float min_y, float max_x, float max_y) { m_bounding_box.set(min_x, min_y, max_x, max_y); }

	// total number of points in the tree node
	__host__ __device__ __forceinline__ int num_points() const { return m_end - m_begin; }

	// the range of points in the tree node
	__host__ __device__ __forceinline__ int points_begin() const { return m_begin; }
	__host__ __device__ __forceinline__ int points_end()   const { return m_end; }

	// define the range of the tree node
	__host__ __device__ __forceinline__ void set_range(int begin, int end) 
	{ 
		m_begin = begin; 
		m_end = end; 
	}

private:

	// identifier
	int m_id;

	// bounding box of the node
	Bounding_box m_bounding_box;

	// the range of points
	int m_begin, m_end;
};

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




    cout << "Closing..." << endl;

    return 0;
}
