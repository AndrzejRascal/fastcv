#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "utils.cuh"

__global__ void laplacianKernel(unsigned char* Pin, unsigned char* Pout, int width, int height) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= height || c >= width) {
        return;
    }

    // [  0,  1,  0 ]
    // [  1, -4,  1 ]
    // [  0,  1,  0 ]
    const int K[3][3] = { { 0, 1, 0 }, { 1, -4, 1 }, { 0, 1, 0 } };

    int sum = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {

            int nr = r + i;
            int nc = c + j;

            if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
                unsigned char I_in = Pin[nr * width + nc];
                sum += I_in * K[i + 1][j + 1];
            }
        }
    }

    int magnitude = abs(sum);

    unsigned char output_value = (magnitude > 255) ? 255 : (unsigned char)magnitude;

    Pout[r * width + c] = output_value;
}

torch::Tensor laplacian(torch::Tensor img){
    TORCH_CHECK(img.device().type() == torch::kCUDA);
    TORCH_CHECK(img.dtype() == torch::kByte);

    img = img.contiguous();

    const auto height = img.size(0);
    const auto width = img.size(1);

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    auto result = torch::empty({height, width},
                               torch::TensorOptions()
                                   .dtype(torch::kByte)
                                   .device(img.device()));

    laplacianKernel<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        width,
        height);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}