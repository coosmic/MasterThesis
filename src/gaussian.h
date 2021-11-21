#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;

Matrix getGaussianKernel(int rows, int cols, double sigmax, double sigmay)
{
    auto gauss_x = Array(cols);

    const auto x_mid = (cols-1) / 2.0;
    const auto y_mid = (rows-1) / 2.0;

    const auto x_spread = 1. / (sigmax*sigmax*2);
    const auto y_spread = 1. / (sigmay*sigmay*2);

    for (auto i = 0;  i < cols;  ++i) {
        auto const x = i - x_mid;
        gauss_x[i] = std::exp(-x*x * x_spread);
    }

    Matrix kernel(rows, Array(cols));
    double sum = 0.0;
    for (auto i = 0;  i < cols;  ++i) {
        for(auto j = 0; j < rows; ++j){
            auto const y = j - y_mid;
            kernel[i][j] = gauss_x[i] * std::exp(-y*y * y_spread);
            sum += kernel[i][j];
        }
    }

    /*sum = 1/sum;
    for (auto i = 0;  i < rows;  ++i) {
        for(auto j = 0; j < cols; ++j){
            auto const y = i - y_mid;
            kernel[j][i] *= sum;
        }
    }*/

    return kernel;
}