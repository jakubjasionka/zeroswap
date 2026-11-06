/*
 * zeroswap_cuda_ext.cu
 * Low-level C++/CUDA bindings for asynchronous HtoD weight streaming (`cudaMemcpyAsync`),
 * batched layer-wise in-place weight overwrite, and cross-stream event synchronization
 * (`cudaEventRecord` + `cudaStreamWaitEvent`) without invalidating CUDA Graph virtual tensor pointers.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <algorithm>

void async_inplace_h2d_overwrite(
    torch::Tensor dst_gpu,
    torch::Tensor src_pinned_cpu,
    int64_t cuda_stream_handle
) {
    TORCH_CHECK(dst_gpu.is_cuda(), "dst_gpu must be a CUDA tensor");
    TORCH_CHECK(!src_pinned_cpu.is_cuda(), "src_pinned_cpu must reside in host memory");
    TORCH_CHECK(dst_gpu.is_contiguous() && src_pinned_cpu.is_contiguous(), "Tensors must be contiguous");

    const size_t copy_bytes = std::min(
        static_cast<size_t>(dst_gpu.numel() * dst_gpu.element_size()),
        static_cast<size_t>(src_pinned_cpu.numel() * src_pinned_cpu.element_size())
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

int64_t async_batch_inplace_h2d_overwrite(
    const std::vector<torch::Tensor>& dst_gpu_list,
    const std::vector<torch::Tensor>& src_pinned_cpu_list,
    int64_t cuda_stream_handle
) {
    TORCH_CHECK(
        dst_gpu_list.size() == src_pinned_cpu_list.size(),
        "dst_gpu_list and src_pinned_cpu_list must have identical length"
    );

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_handle);
    size_t total_bytes = 0;

    for (size_t i = 0; i < dst_gpu_list.size(); ++i) {
        const auto& dst = dst_gpu_list[i];
        const auto& src = src_pinned_cpu_list[i];
        TORCH_CHECK(dst.is_cuda(), "dst tensor at index ", i, " must be on CUDA");
        TORCH_CHECK(!src.is_cuda(), "src tensor at index ", i, " must be on Host");
        TORCH_CHECK(dst.is_contiguous() && src.is_contiguous(), "Tensors at index ", i, " must be contiguous");

        const size_t copy_bytes = std::min(
            static_cast<size_t>(dst.numel() * dst.element_size()),
            static_cast<size_t>(src.numel() * src.element_size())
        );

        cudaError_t err = cudaMemcpyAsync(
            dst.data_ptr(),
            src.data_ptr(),
            copy_bytes,
            cudaMemcpyHostToDevice,
            stream
        );
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed at index ", i, ": ", cudaGetErrorString(err));
        total_bytes += copy_bytes;
    }

    return static_cast<int64_t>(total_bytes);
}

int64_t create_stream_barrier_event(int64_t copy_stream_handle) {
    cudaStream_t copy_stream = reinterpret_cast<cudaStream_t>(copy_stream_handle);
    cudaEvent_t ev;
    cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    TORCH_CHECK(err == cudaSuccess, "cudaEventCreateWithFlags failed: ", cudaGetErrorString(err));
    err = cudaEventRecord(ev, copy_stream);
    TORCH_CHECK(err == cudaSuccess, "cudaEventRecord failed: ", cudaGetErrorString(err));
    return reinterpret_cast<int64_t>(ev);
}

void wait_and_destroy_barrier_event(int64_t compute_stream_handle, int64_t event_handle) {
    cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(compute_stream_handle);
    cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(event_handle);
    cudaError_t err = cudaStreamWaitEvent(compute_stream, ev, 0);
    TORCH_CHECK(err == cudaSuccess, "cudaStreamWaitEvent failed: ", cudaGetErrorString(err));
    err = cudaEventDestroy(ev);
    TORCH_CHECK(err == cudaSuccess, "cudaEventDestroy failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "async_inplace_h2d_overwrite",
        &async_inplace_h2d_overwrite,
        "Zero-copy asynchronous Host-to-Device in-place weight overwrite on a dedicated CUDA stream"
    );
    m.def(
        "async_batch_inplace_h2d_overwrite",
        &async_batch_inplace_h2d_overwrite,
        "Batched zero-copy asynchronous Host-to-Device in-place weight overwrite for a full transformer layer"
    );
    m.def(
        "create_stream_barrier_event",
        &create_stream_barrier_event,
        "Creates and records a zero-overhead CUDA event on the DMA copy stream"
    );
    m.def(
        "wait_and_destroy_barrier_event",
        &wait_and_destroy_barrier_event,
        "Waits on a CUDA event from the compute stream via cudaStreamWaitEvent and destroys the handle"
    );
}
