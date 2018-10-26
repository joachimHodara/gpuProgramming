#include <cassert>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

/**
 * Flip the image in the vertical direction
 */
void flipImageV(Mat3b& img)
{
  const size_t nrows = img.rows;
  const size_t ncols = img.cols;
  for(size_t i = 0; i < nrows/2; ++i) {
    for(size_t j = 0; j < ncols; ++j) {
      Vec3b temp_pixel{img(i,j)};
      img(i,j) = img(nrows-i-1,j);
      img(nrows-i-1,j) = temp_pixel;
    }
  }
}

/**
 * Flip the image in the horizontal direction
 */
void flipImageH(Mat3b& img)
{
  const size_t nrows = img.rows;
  const size_t ncols = img.cols;
  for(size_t i = 0; i < nrows; ++i) {
    for(size_t j = 0; j < ncols/2; ++j) {
      Vec3b temp_pixel{img(i,j)};
      img(i,j) = img(i,ncols-j-1);
      img(i,ncols-j-1) = temp_pixel;
    }
  }
}

/**
 * Flips an image in the direction specified by the user.
 * This function takes 3 arguments:
 * @param argv[1] is the image filename
 * @param argv[2] is the flip direction ('v' or 'h')
 * @param argv[3] is the number of times the function gets repeated (for better timing)
 */
int main(int argc, char* argv[])
{
  // Load the image
  Mat3b image = imread(argv[1]);
  if(image.empty()) {
    cerr << "Could not read image" << endl;
    return 1;
  }
  namedWindow("Source", WINDOW_AUTOSIZE);
  imshow("Source", image);

  // number of times the function will be computed (must be an odd number or nothing happens)
  const size_t num_repet = atoi(argv[3]);
  assert(num_repet % 2 != 0);

  auto start = std::chrono::high_resolution_clock::now();

  // Rotate the image depending on the user input
  switch(argv[2][0]) {
    case 'v': for(size_t it = 0; it < num_repet; ++it) flipImageV(image); break;
    case 'h': for(size_t it = 0; it < num_repet; ++it) flipImageH(image); break;
    default: cerr << "Invalid argument to main" << endl;
  }

  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = stop - start;
  cout << "Elapsed time per call: " << 1000.0*elapsed_time.count()/num_repet << "ms" << endl;

  // display the final image
  namedWindow("Destination", WINDOW_AUTOSIZE);
  imshow("Destination", image);
  waitKey(0);
  cout << "Closing ..." << endl;

  return 0;
}

