#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Softmax kernels (declared in softmax/softmax_kernels.cuh)
// ---------------------------------------------------------------------------
__global__ void SoftmaxNaiveKernel(const float* x, float* y, int rows, int cols);
__global__ void SoftmaxV1Kernel(const float* __restrict__ x, float* __restrict__ y,
                                int rows, int cols);
__global__ void SoftmaxV2Kernel(const float* __restrict__ x, float* __restrict__ y,
                                int rows, int cols);
__global__ void SoftmaxV3Kernel(const float* __restrict__ x, float* __restrict__ y,
                                int rows, int cols);

// ---------------------------------------------------------------------------
// RMSNorm kernels (declared in rmsnorm/rmsnorm_kernels.cuh)
// ---------------------------------------------------------------------------
__global__ void RMSNormV0Kernel(const float* x, float* y, const float* weight,
                                int rows, int cols, float eps);
__global__ void RMSNormV1Kernel(const float* __restrict__ x, float* __restrict__ y,
                                const float* __restrict__ weight, int rows, int cols,
                                float eps);
__global__ void RMSNormV2Kernel(const float* __restrict__ x, float* __restrict__ y,
                                const float* __restrict__ weight, int rows, int cols,
                                float eps);
__global__ void RMSNormV3Kernel(const float* __restrict__ x, float* __restrict__ y,
                                const float* __restrict__ weight, int rows, int cols,
                                float eps);

// ---------------------------------------------------------------------------
// Softmax launcher
// ---------------------------------------------------------------------------
torch::Tensor softmax_forward_cuda(torch::Tensor x, int version) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Only float32 is supported");

    int rows = x.size(0);
    int cols = x.size(1);
    auto y = torch::empty_like(x);

    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;

    if (version == 0) {
        // Naive: 1 thread per row
        int threads = 256;
        int blocks = (rows + threads - 1) / threads;
        SoftmaxNaiveKernel<<<blocks, threads>>>(x_ptr, y_ptr, rows, cols);
    } else if (version == 1) {
        // V1: 1 warp per row, serial reduce
        int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        size_t smem = WARPS_PER_BLOCK * cols * sizeof(float);
        SoftmaxV1Kernel<<<blocks, BLOCK_SIZE, smem>>>(x_ptr, y_ptr, rows, cols);
    } else if (version == 2) {
        // V2: warp shuffle reduce
        int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        size_t smem = WARPS_PER_BLOCK * cols * sizeof(float);
        SoftmaxV2Kernel<<<blocks, BLOCK_SIZE, smem>>>(x_ptr, y_ptr, rows, cols);
    } else if (version == 3) {
        // V3: online softmax
        int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        size_t smem = WARPS_PER_BLOCK * cols * sizeof(float);
        SoftmaxV3Kernel<<<blocks, BLOCK_SIZE, smem>>>(x_ptr, y_ptr, rows, cols);
    } else {
        TORCH_CHECK(false, "version must be 0, 1, 2, or 3");
    }

    return y;
}

// ---------------------------------------------------------------------------
// RMSNorm launcher
// ---------------------------------------------------------------------------
torch::Tensor rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor weight, float eps,
                                   int version) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(x.size(1) == weight.size(0), "x.cols must match weight.size");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Only float32 is supported");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Only float32 is supported");

    int rows = x.size(0);
    int cols = x.size(1);
    auto y = torch::empty_like(x);

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;

    if (version == 0) {
        // Naive: 1 thread per row
        int threads = 256;
        int blocks = (rows + threads - 1) / threads;
        RMSNormV0Kernel<<<blocks, threads>>>(x_ptr, y_ptr, w_ptr, rows, cols, eps);
    } else if (version == 1) {
        // V1: 1 warp per row, serial reduce
        int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        size_t smem = WARPS_PER_BLOCK * (cols + 1) * sizeof(float);
        RMSNormV1Kernel<<<blocks, BLOCK_SIZE, smem>>>(x_ptr, y_ptr, w_ptr, rows, cols, eps);
    } else if (version == 2) {
        // V2: warp shuffle reduce
        int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        size_t smem = WARPS_PER_BLOCK * (cols + 1) * sizeof(float);
        RMSNormV2Kernel<<<blocks, BLOCK_SIZE, smem>>>(x_ptr, y_ptr, w_ptr, rows, cols, eps);
    } else if (version == 3) {
        // V3: weight in smem, no staging
        int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        size_t smem = cols * sizeof(float);
        RMSNormV3Kernel<<<blocks, BLOCK_SIZE, smem>>>(x_ptr, y_ptr, w_ptr, rows, cols, eps);
    } else {
        TORCH_CHECK(false, "version must be 0, 1, 2, or 3");
    }

    return y;
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_forward", &softmax_forward_cuda, "Softmax forward (CUDA)",
          py::arg("x"), py::arg("version") = 3);
    m.def("rmsnorm_forward", &rmsnorm_forward_cuda, "RMSNorm forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-5f,
          py::arg("version") = 3);
}

// ---------------------------------------------------------------------------
// TORCH_LIBRARY registration (optional, for torch.ops style usage)
// ---------------------------------------------------------------------------
TORCH_LIBRARY(cuda_kernel_ops, m) {
    m.def("softmax_forward(Tensor x, int version=3) -> Tensor");
    m.def("rmsnorm_forward(Tensor x, Tensor weight, float eps=1e-5, int version=3) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_kernel_ops, CUDA, m) {
    m.impl("softmax_forward", softmax_forward_cuda);
    m.impl("rmsnorm_forward", rmsnorm_forward_cuda);
}
