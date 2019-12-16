#include <iostream>
#include <fstream>
#include <mpi.h>

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#define RESOLUTION 1000
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define ITERATIONS 1000 // Maximum number of iterations

#define MASTER 0

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int nproc, myid;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    int *image = nullptr;
    if (myid == MASTER)
        image = new int[HEIGHT * WIDTH];

    if (HEIGHT * WIDTH % nproc != 0)
    {
        if (myid == 0)
            cout << "It's not possible to split the task in " << nproc << " nodes." << endl;
        MPI_Abort(MPI_COMM_WORLD, -3);
    }

    int slice_size = HEIGHT * WIDTH / nproc;
    int *const image_slice = new int[slice_size];

    double start = MPI_Wtime();

    for (int pos = 0; pos < slice_size; pos++)
        image_slice[pos] = 0;

    int absolute_start_idx = slice_size * myid;
#pragma omp parallel for default(none) shared(image_slice, absolute_start_idx, slice_size) schedule(dynamic)
    for (int pos = 0; pos < slice_size; pos++)
    {
        int absolute_idx = absolute_start_idx + pos;
        int row = absolute_idx / WIDTH;
        int col = absolute_idx % WIDTH;

        double c_real = col * STEP + MIN_X;
        double c_im = row * STEP + MIN_Y;

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
                image_slice[pos] = i;
                break;
            }
        }
    }

    MPI_Gather(image_slice, slice_size, MPI_INT, image, slice_size, MPI_INT, MASTER, MPI_COMM_WORLD);
    double end = MPI_Wtime();

    delete[] image_slice;

    if (myid == MASTER)
    {
        cout << "Time elapsed: "
             << (int)(end - start)
             << " seconds." << endl;

        // Write the result to a file
        ofstream matrix_out;

        if (argc < 2)
        {
            cout << "Please specify the output file as a parameter." << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        matrix_out.open(argv[1], ios::trunc);
        if (!matrix_out.is_open())
        {
            cout << "Unable to open file." << endl;
            MPI_Abort(MPI_COMM_WORLD, -2);
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

        // It's here for coding style, but useless
        delete[] image;
    }

    MPI_Finalize();
    return 0;
}