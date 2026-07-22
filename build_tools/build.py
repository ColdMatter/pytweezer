import sys
import numpy as np
from pybind11.setup_helpers import Pybind11Extension, build_ext

if sys.platform == "win32":
    # Windows (MSVC)
    # /GL = Link-Time Optimization (Compile)
    # /LTCG = Link-Time Optimization (Link)
    tophat_compile_args = ["/O2", "/openmp:llvm", "/GL"]
    tophat_link_args = ["/LTCG"]
    
    sum_compile_args = ["/O2", "/GL"]
    sum_link_args = ["/LTCG"]
else:
    # Linux/macOS (GCC/Clang)
    # -flto = Link-Time Optimization
    tophat_compile_args = ["-O3", "-fopenmp", "-flto"]
    tophat_link_args = ["-fopenmp", "-flto"]
    
    sum_compile_args = ["-O3", "-flto"]
    sum_link_args = ["-flto"]

# extensions
ext_modules = [
    # Morphological Top Hat
    Pybind11Extension(
        "pytweezer.cpp.morph_tophat_cpp",
        ["cpp_implementations/Morph_top_hat/Morphological_top_hat_cpp.cpp"],
        cxx_std=17, 
        extra_compile_args=tophat_compile_args,
        extra_link_args=tophat_link_args,
    ),
    
    #  Sum Pixel Values
    Pybind11Extension(
        "pytweezer.cpp.sum_pixel_values_cpp",
        ["cpp_implementations/Pixel_Summing/sum_pixel_values.cpp"],
        cxx_std=14, 
        include_dirs=[np.get_include()],
        extra_compile_args=sum_compile_args,
        extra_link_args=sum_link_args,
    )
]

#  for Poetry
def build(setup_kwargs):
    setup_kwargs.update({
        "ext_modules": ext_modules,
        "cmdclass": {"build_ext": build_ext},
        "zip_safe": False,
    })