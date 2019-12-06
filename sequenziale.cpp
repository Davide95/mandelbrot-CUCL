#include <iostream>
#include <complex>
#include <SFML/Graphics.hpp>

#define DEBUG

#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

#define RESOLUTION 10
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2
#define ITERATIONS 1000

using namespace std;

int main() {
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
    for(int row = 0; row < HEIGHT; row++) {
        for(int col = 0; col < WIDTH; col++) {
            cout << image[row][col] << "\t";
        }
        cout << endl;
    }
#endif

    return 0;
}