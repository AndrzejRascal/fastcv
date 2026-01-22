#include <cstdio>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <stdio.h>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

void get_size(int* w, int* h) {
    cv::Mat image = cv::imread("/content/kocia.jpg");
     if (image.empty()) {
        *w = 0;
        *h = 0;
        std::cout << "image is empty";
        return;
     }
    *w = image.size().width;
    *h = image.size().height;

    cv::imwrite("/content/out.jpg", image);
    cv::waitKey(0);
    return;

}

template<typename T>
__global__ void laplacian_kernel(T* input, T* output, int width, int height) {

    __shared__ float tile[18][18];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    if (x < width && y < height) {
        tile[ty][tx] = (float)input[y * width + x];

        if (threadIdx.x == 0 && x > 0) tile[ty][0] = (float)input[y * width + (x - 1)];
        if (threadIdx.x == blockDim.x - 1 && x < width - 1) tile[ty][17] = (float)input[y * width + (x + 1)];
        if (threadIdx.y == 0 && y > 0) tile[0][tx] = (float)input[(y - 1) * width + x];
        if (threadIdx.y == blockDim.y - 1 && y < height - 1) tile[17][tx] = (float)input[(y + 1) * width + x];
    }

    __syncthreads();

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float res = tile[ty - 1][tx] + tile[ty + 1][tx] +
                    tile[ty][tx - 1] + tile[ty][tx + 1] -
                    4.0f * tile[ty][tx];

        output[y * width + x] = (T)(res + 128.0f);
    }
}

int laplacian() {
    int w, h;
    get_size(&w, &h);

    thrust::host_vector<float> h_input(w * h, 1.0f);

    thrust::device_vector<float> d_input = h_input;
    thrust::device_vector<float> d_output(w * h);

    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    laplacian_kernel<float><<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_output.data()),
        w, h
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("launch error: %s\n", cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("sync error: %s\n", cudaGetErrorString(err));

    thrust::host_vector<float> h_output = d_output;
    return 0;
}
