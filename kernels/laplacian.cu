#include <cstdio>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/pair.h>

#include <cub/cub.cuh>

#include <utility>
#include <tuple>

#include "utils.cuh"

#include "nvtx3.hpp"

using u8 = unsigned char;

static inline thrust::pair<int, int> image_hw(const torch::Tensor& img)
{
    return thrust::make_pair((int)img.size(0), (int)img.size(1));
}

template<typename T>
__host__ __device__ __forceinline__ T clamp_value(T v, T lo, T hi)
{
    return (v < lo) ? lo : (v > hi ? hi : v);
}

struct Clamp
{
    __host__ __device__ u8 operator()(int v) const
    {
        v = clamp_value<int>(v, 0, 255);
        return (u8)v;
    }
};

__global__ void laplacianKernel(const u8* in, int* out_sum, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    bool inside_image = (row < height && col < width);

    int block_width  = blockDim.x;
    int block_height = blockDim.y;

    int shared_width  = block_width  + 2;
    int shared_height = block_height + 2;

    extern __shared__ u8 shared_tile[];

    int thread_id    = threadIdx.y * block_width + threadIdx.x;
    int threads_num  = block_width * block_height;
    int shared_count = shared_width * shared_height;

    for (int shared_id = thread_id; shared_id < shared_count; shared_id += threads_num)
    {
        int tile_row = shared_id / shared_width;
        int tile_col = shared_id - tile_row * shared_width;

        int src_col = (int)(blockIdx.x * block_width)  + (tile_col - 1);
        int src_row = (int)(blockIdx.y * block_height) + (tile_row - 1);

        u8 pixel = 0;
        if (src_col >= 0 && src_col < width && src_row >= 0 && src_row < height)
            pixel = in[src_row * width + src_col];

        shared_tile[shared_id] = pixel;
    }

    __syncthreads();

    if (!inside_image) return;

    int tile_col = threadIdx.x + 1;
    int tile_row = threadIdx.y + 1;

    int center = (int)shared_tile[ tile_row * shared_width + tile_col];
    int up = (int)shared_tile[(tile_row - 1) * shared_width + tile_col];
    int down = (int)shared_tile[(tile_row + 1) * shared_width + tile_col];
    int left = (int)shared_tile[ tile_row * shared_width + (tile_col - 1)];
    int right = (int)shared_tile[ tile_row * shared_width + (tile_col + 1)];

    int sum = -4 * center + up + down + left + right;

    out_sum[row * width + col] = sum;
}

torch::Tensor laplacian(torch::Tensor img)
{
    TORCH_CHECK(img.device().type() == torch::kCUDA);
    TORCH_CHECK(img.dtype() == torch::kByte);

    img = img.contiguous();

    auto [height, width] = image_hw(img);
    const int n = height * width;

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    auto result = torch::empty({height, width},
                               torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    auto sum_tensor = torch::empty({height, width},
                                   torch::TensorOptions().dtype(torch::kInt32).device(img.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t cuda_stream = stream.stream();

    size_t shared_bytes = (dimBlock.x + 2) * (dimBlock.y + 2) * sizeof(u8);

    laplacianKernel<<<dimGrid, dimBlock, shared_bytes, cuda_stream>>>(
        img.data_ptr<u8>(),
        sum_tensor.data_ptr<int>(),
        width,
        height);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto sum_dev = thrust::device_pointer_cast(sum_tensor.data_ptr<int>());
    auto out_dev = thrust::device_pointer_cast(result.data_ptr<u8>());

    auto fancy_begin = thrust::make_transform_iterator(sum_dev, Clamp{});
    auto fancy_end   = fancy_begin + n;

    auto exec_space = thrust::cuda::par.on(cuda_stream);
    thrust::copy(exec_space, fancy_begin, fancy_end, out_dev);

    return result;
}