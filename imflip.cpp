#include <cassert>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

using namespace cv;
using namespace std;

/**
 * Flip the image in the vertical direction
 */
inline void flipImageV(size_t nthreads, size_t tid, Mat3b& img)
{
  const size_t nrows = img.rows;
  const size_t ncols = img.cols;

  const size_t cols_per_thread = ceil(float(ncols)/nthreads);
  const size_t colb = tid*cols_per_thread;
  const size_t cole = min(colb + cols_per_thread, ncols);

  for(size_t i = 0; i < nrows/2; ++i) {
    for(size_t j = colb; j < cole; ++j) {
      Vec3b temp_pixel{img(i,j)};
      img(i,j) = img(nrows-i-1,j);
      img(nrows-i-1,j) = temp_pixel;
    }
  }
}

/**
 * Flip the image in the horizontal direction
 */
inline void flipImageH(size_t nthreads, size_t tid, Mat3b& img)
{
  const size_t nrows = img.rows;
  const size_t ncols = img.cols;

  const size_t rows_per_thread = ceil(float(nrows)/nthreads);
  const size_t rowb = tid*rows_per_thread;
  const size_t rowe = min(rowb + rows_per_thread, nrows);

  for(size_t i = rowb; i < rowe; ++i) {
    for(size_t j = 0; j < ncols/2; ++j) {
      Vec3b temp_pixel{img(i,j)};
      img(i,j) = img(i,ncols-j-1);
      img(i,ncols-j-1) = temp_pixel;
    }
  }
}

/**
 * Dispatch to the right function based on user input
 */
void flipImage(Mat3b& image, size_t nthreads, size_t tid, size_t number_repet, char flip_direction)
{
  switch(flip_direction) {
    case 'v': for(size_t it=0; it<number_repet; ++it) flipImageV(nthreads, tid, image); break;
    case 'h': for(size_t it=0; it<number_repet; ++it) flipImageH(nthreads, tid, image); break;
    default: cerr << "Invalid argument to main" << endl;
  }
}

/**
 * Flips an image in the direction specified by the user.
 * This function takes 3 or 4 arguments:
 * @param argv[1] is the image filename
 * @param argv[2] is the flip direction ('v' or 'h')
 * @param argv[3] is the number of times the function gets repeated (for better timing)
 * @param argv[4] (optional) number of threads
 */
int main(int argc, char* argv[])
{
  // Load the image
  Mat3b image = imread(argv[1]);
  if(image.empty()) {
    cerr << "Could not read image" << endl;
    return 1;
  }
//  namedWindow("Source", WINDOW_AUTOSIZE);
//  imshow("Source", image);

  // Read-in the flip direction
  char flip_direction = argv[2][0];

  // number of times the function will be computed (must be an odd number or nothing happens)
  const size_t num_repet = atoi(argv[3]);
  assert(num_repet % 2 != 0);

  // Read-in the number of threads
  int threads_number = 1;
  if(argc == 5) {
    threads_number = atoi(argv[4]);
    cout << "Executing in parallel using " << threads_number << " threads." << endl;
  }
  else
    cout << "Executing in serial..." << endl;

  auto start = std::chrono::high_resolution_clock::now();

  // Rotate the image depending on the user input
  vector<thread> threads(threads_number);
  for(size_t tid = 0; tid < threads_number; ++tid) 
    threads[tid] = thread(flipImage, std::ref(image), threads_number, tid, num_repet, flip_direction);

  cout << "The " << threads_number << " threads have been unleashed!" << endl;

  for(size_t tid = 0; tid < threads_number; ++tid) 
    threads[tid].join();

  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = stop - start;
  cout << "Elapsed time per call: " << 1000.0*elapsed_time.count()/num_repet << "ms" << endl;

  // display the final image
//  namedWindow("Destination", WINDOW_AUTOSIZE);
//  imshow("Destination", image);
//  waitKey(0);
  cout << "Closing ..." << endl;

  return 0;
}

