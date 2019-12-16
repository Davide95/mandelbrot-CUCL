#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <omp.h>

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
#define RESOLUTION 12000
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define ITERATIONS 2000 // Maximum number of iterations

// GPU stuff
#define THREADS 8
#define N_GPUS 2
#define GPU_IMG_SIZE_IDX ((HEIGHT/N_GPUS) * WIDTH)
#define GPU_IMG_SIZE (GPU_IMG_SIZE_IDX * sizeof(int))

using namespace std;

__global__ void computePixel(int* image, int dev_id)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < HEIGHT/N_GPUS && col < WIDTH) {
		const double c_real = col * STEP + MIN_X;
		const double c_im = (row + dev_id*(HEIGHT/N_GPUS)) * STEP + MIN_Y;

		// z = z^2 + c
		double z_real = 0, z_im = 0;		
		for (int i = 1; i <= ITERATIONS; i++)
		{
			double tmp = (z_real * z_real - z_im * z_im) + c_real;
			z_im = (2 * z_real * z_im) + c_im;
			z_real = tmp;

			// If it is convergent
			if (z_real * z_real + z_im * z_im >= 4)
			{
				image[row * WIDTH + col] = i;
				return;
			}
		}
	}
}

int main(int argc, char** argv)
{
	int* const image = new int[HEIGHT * WIDTH];

	if(HEIGHT % N_GPUS != 0) {
		cout << "The height of the image (" << HEIGHT << ") is not divisible by the number of GPUs (" << N_GPUS << ")" << endl;
		return -3;
	}

	int *image_on_gpus[N_GPUS];
	#pragma omp parallel num_threads(N_GPUS)
	{
		int dev_id = omp_get_thread_num();
		cudaSetDevice(dev_id);
		cudaMalloc(&image_on_gpus[dev_id], GPU_IMG_SIZE);
    }

    const dim3 threads(THREADS, THREADS);
    const int BLOCKS_X = ((WIDTH + threads.x - 1) / threads.x);
    const int BLOCKS_Y = ((HEIGHT/N_GPUS + threads.y - 1) / threads.y);
    const dim3 blocks(BLOCKS_X, BLOCKS_Y);

	const auto start = chrono::steady_clock::now();
    
	for (int pos = 0; pos < HEIGHT * WIDTH; pos++)
		image[pos] = 0;

    #pragma omp parallel num_threads(N_GPUS)
    {
        int dev_id = omp_get_thread_num();
		cudaSetDevice(dev_id);
        cudaMemcpy(image_on_gpus[dev_id], &image[dev_id * GPU_IMG_SIZE_IDX], GPU_IMG_SIZE, cudaMemcpyHostToDevice);
	    computePixel << < blocks, threads >> > (image_on_gpus[dev_id], dev_id);
        cudaDeviceSynchronize();
    }
    
	const auto end = chrono::steady_clock::now();

	cout << "Time elapsed: "
		<< chrono::duration_cast<chrono::seconds>(end - start).count()
		<< " seconds." << endl;
	
    #pragma omp parallel num_threads(N_GPUS)
    {
        int dev_id = omp_get_thread_num();
		cudaSetDevice(dev_id);
	    cudaMemcpy(&image[dev_id * GPU_IMG_SIZE_IDX], image_on_gpus[dev_id], GPU_IMG_SIZE, cudaMemcpyDeviceToHost);
	    cudaFree(image_on_gpus[dev_id]);
    }

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
