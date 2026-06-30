// bindings.cpp -- pybind11 wrapper exposing the white-tophat filter (only) as
// the compiled module `fast_tophat_filter`.
//
//   fast_tophat_filter.white_tophat(image, size)        -> filtered image
//   fast_tophat_filter.batch_white_tophat(stack, size)  -> filtered stack
//
// Drop-in for scipy.ndimage.white_tophat(image, size=N, mode='reflect')
// (and the project's morphological_tophat_high_pass), ~2x faster, bit-identical.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "tophat_filter.hpp"

namespace py = pybind11;

// White tophat on a single 2-D float32/float64 image.
template <typename T>
py::array_t<T> white_tophat_py(py::array_t<T, py::array::c_style | py::array::forcecast> img,
                               int size) {
    auto buf = img.request();
    if (buf.ndim != 2) throw std::runtime_error("expected a 2-D image");
    int h = static_cast<int>(buf.shape[0]);
    int w = static_cast<int>(buf.shape[1]);
    auto out = py::array_t<T>({h, w});
    tophat::white_tophat<T>(static_cast<const T*>(buf.ptr),
                            static_cast<T*>(out.request().ptr), h, w, size);
    return out;
}

// White tophat over a 3-D (n, H, W) stack -- one call, no Python loop. This was to test the difference in overhead 
// between a single call to the C++ function vs. a Python loop calling the single-image function n times.
template <typename T>
py::array_t<T> batch_white_tophat_py(py::array_t<T, py::array::c_style | py::array::forcecast> stack,
                                     int size) {
    auto buf = stack.request();
    if (buf.ndim != 3) throw std::runtime_error("expected a 3-D (n, H, W) stack");
    int n = static_cast<int>(buf.shape[0]);
    int h = static_cast<int>(buf.shape[1]);
    int w = static_cast<int>(buf.shape[2]);
    auto out = py::array_t<T>({n, h, w});
    const T* in = static_cast<const T*>(buf.ptr);
    T* op = static_cast<T*>(out.request().ptr);
    tophat::batch_white_tophat<T>(in, op, n, h, w, size);
    return out;
}

PYBIND11_MODULE(fast_tophat_filter, m) {
    m.doc() = "Compiled C++ white-tophat high-pass filter for tweezer arrays "
              "(filter only; bit-identical to scipy white_tophat, ~2x faster).";

    m.def("white_tophat", &white_tophat_py<float>, py::arg("image"), py::arg("size"));
    m.def("white_tophat", &white_tophat_py<double>, py::arg("image"), py::arg("size"));

    m.def("batch_white_tophat", &batch_white_tophat_py<float>, py::arg("stack"), py::arg("size"));
    m.def("batch_white_tophat", &batch_white_tophat_py<double>, py::arg("stack"), py::arg("size"));
}
