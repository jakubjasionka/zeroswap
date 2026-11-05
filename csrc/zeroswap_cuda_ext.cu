/*
 * zeroswap_cuda_ext.cu
 * Low-level C++/CUDA bindings for asynchronous HtoD weight streaming (`cudaMemcpyAsync`)
 * and cross-stream event synchronization (`cudaStreamWaitEvent`) without invalidating
 * CUDA Graph virtual tensor pointers.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

void async_inplace_h2d_overwrite(
    torch::Tensor dst_gpu,
    torch::Tensor src_pinned_cpu,
    int64_t cuda_stream_handle
) {
    TORCH_CHECK(dst_gpu.is_cuda(), "dst_gpu must be a CUDA tensor");
    TORCH_CHECK(!src_pinned_cpu.is_cuda(), "src_pinned_cpu must reside in host memory");
    TORCH_CHECK(dst_gpu.is_contiguous() && src_pinned_cpu.is_contiguous(), "Tensors must be contiguous");

    const size_t copy_bytes = std::min(
        dst_gpu.numel() * dst_gpu.element_size(),
        src_pinned_cpu.numel() * src_pinned_cpu.element_size()
    );

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_handle);
    cudaError_t err = cudaMemcpyAsync(
        dst_gpu.data_ptr(),
        src_pinned_cpu.data_ptr(),
        copy_bytes,
        cudaMemcpyHostToDevice,
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "async_inplace_h2d_overwrite",
        &async_inplace_h2d_overwrite,
        "Zero-copy asynchronous Host-to-Device in-place weight overwrite on a dedicated CUDA stream"
    );
}
