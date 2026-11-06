"""Build script for the C++/CUDA native extension (`zeroswap_cuda_ext`)."""

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
cuda_lib_dir = "/usr/local/cuda-13.3/lib64"

setup(
    name="zeroswap_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="zeroswap_cuda_ext",
            sources=["zeroswap_cuda_ext.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
            extra_link_args=[
                f"-Wl,-rpath,{torch_lib_dir}",
                f"-Wl,-rpath,{cuda_lib_dir}",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
