#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>

#include <iostream>
#include <fstream>
#include <chrono>

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#define RESOLUTION 3000
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define ITERATIONS 2000 // Maximum number of iterations

// GPU stuff
#define THREADS 32
#define GPU_IMG_SIZE (HEIGHT * WIDTH * sizeof(int))

using namespace std;

__global__ void computePixel(int *image)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < HEIGHT && col < WIDTH)
    {
        const cuDoubleComplex c = make_cuDoubleComplex(col * STEP + MIN_X, row * STEP + MIN_Y);

        // z = z^2 + c
        cuDoubleComplex z = make_cuDoubleComplex(0, 0);
        for (int i = 1; i <= ITERATIONS; i++)
        {
            z = cuCadd(cuCmul(z, z), c);

            // If it is divergent
            if (cuCabs(z) >= 2)
            {
                image[row * WIDTH + col] = i;
                return;
            }
        }
    }
}

int main(int argc, char **argv)
{
    int *const image = new int[HEIGHT * WIDTH];
    int *image_on_gpu;
    cudaMalloc(&image_on_gpu, GPU_IMG_SIZE);

    const dim3 threads(THREADS, THREADS);
    const int BLOCKS_X = ((WIDTH + threads.x - 1) / threads.x);
    const int BLOCKS_Y = ((HEIGHT + threads.y - 1) / threads.y);
    const dim3 blocks(BLOCKS_X, BLOCKS_Y);

    const auto start = chrono::steady_clock::now();

    for (int pos = 0; pos < HEIGHT * WIDTH; pos++)
        image[pos] = 0;

    cudaMemcpy(image_on_gpu, image, GPU_IMG_SIZE, cudaMemcpyHostToDevice);
    computePixel<<<blocks, threads>>>(image_on_gpu);

    cudaDeviceSynchronize();
    const auto end = chrono::steady_clock::now();

    cout << "Time elapsed: "
         << chrono::duration_cast<chrono::seconds>(end - start).count()
         << " seconds." << endl;

    cudaMemcpy(image, image_on_gpu, GPU_IMG_SIZE, cudaMemcpyDeviceToHost);
    cudaFree(image_on_gpu);

    // Write the result to a file
    ofstream matrix_out;

    if (argc < 2)
    {
        cout << "Please specify the output file as a parameter." << endl;
        return -1;
    }

    matrix_out.open(argv[1], ios::trunc);
    if (!matrix_out.is_open())
    {
        cout << "Unable to open file." << endl;
        return -2;
    }

    for (int row = 0; row < HEIGHT; row++)
    {
        for (int col = 0; col < WIDTH; col++)
        {
            matrix_out << image[row * WIDTH + col];

            if (col < WIDTH - 1)
                matrix_out << ',';
        }
        if (row < HEIGHT - 1)
            matrix_out << endl;
    }
    matrix_out.close();

    delete[] image; // It's here for coding style, but useless
    return 0;
}