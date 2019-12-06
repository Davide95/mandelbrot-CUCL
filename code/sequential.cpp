#include <iostream>
#include <fstream>
#include <complex>

// Uncomment this if you want to use in production
#define DEBUG

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#define RESOLUTION 10
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2 // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations

using namespace std;

int main(int argc, char **argv) {
    int image[HEIGHT][WIDTH];

    for(int row = 0; row < HEIGHT; row++) {
        for(int col = 0; col < WIDTH; col++) {
            image[row][col] = 0;

            // z = z^2 + c
            complex<double> z(0, 0);
            complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);
            for(int i = 0; i < ITERATIONS; i++) {
                z = pow(z, 2) + c;

                // If it is convergent
                if(abs(z) >= 2) {
                    image[row][col] = 1;
                    break;
                }
            }
        }
    }

#ifdef DEBUG
    if(argc < 2) {
        cout << "Please specify the output file as a parameter." << endl;
        return -1;
    }

    ofstream matrix_out;
    matrix_out.open(argv[1], ios::trunc);
    if(!matrix_out.is_open()) {
        cout << "Unable to open file." << endl;
        return -2;
    }

    // Write the result to a file
    for(int row = 0; row < HEIGHT; row++) {
        for(int col = 0; col < WIDTH; col++) {
            matrix_out << image[row][col];

            if(col < WIDTH -1)
                matrix_out << ',';
        }
        if(row < HEIGHT - 1)
            matrix_out << endl;
    }

    matrix_out.close();
#endif

    return 0;
}