"""Optional build script for the C++/CUDA native extension (`zeroswap_cuda_ext`)."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="zeroswap_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="zeroswap_cuda_ext",
            sources=["zeroswap_cuda_ext.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
