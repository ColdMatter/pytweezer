from setuptools import setup, Extension
import pybind11
import numpy as np

ext_modules = [
    Extension(
        "sum_pixel_values_cpp", 
        ["sum_pixel_values.cpp"], 
        include_dirs=[
            pybind11.get_include(),
            np.get_include() 
        ],
        extra_compile_args=["/O2"], # this enables optimisation for MSVC compiler, same as -02 flag in clang 
        language="c++", 
    )
]

setup(
    name="sum_pixel_values_cpp",
    version="0.1",
    ext_modules=ext_modules,
)