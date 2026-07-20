// Morphological white top-hat filter — parallel C++ drop-in for
// scipy.ndimage.white_tophat(image, size=feature_size).
//
// Bit-exact with scipy for flat rectangular structuring elements and the scipy
// defaults mode='reflect', origin=0. Two details are load-bearing and easy to
// get wrong when porting:
//
//   1. A scalar `size` applies the kernel to EVERY axis. For a (256,256,4) RGBA
//      image, size=10 means a 10x10x10 element that also spans the 4-deep
//      channel axis — it is NOT a per-channel 2D filter.
//   2. grey_dilation mirrors the structuring element, so its window is the
//      point reflection of grey_erosion's through 0: erosion spans
//      [-k/2, k - k/2 - 1] and dilation spans [-(k - k/2 - 1), k/2]. These
//      differ whenever k is even — for k=10, [-5,+4] vs [-4,+5]. Reusing one
//      window for both is off-by-one and silently diverges.
//
// Speed comes from three places:
//   * A flat box is separable, so an N-D min/max becomes N 1-D passes.
//   * Each 1-D pass uses van Herk / Gil-Werman: ~3 comparisons per pixel
//     regardless of kernel width, instead of O(k).
//   * OpenMP over independent lines (single image) or over images (batch).
//
// tophat_batch_mean additionally fuses the whole notebook pipeline
// (per-image top-hat -> stack -> np.mean) into one pass, which avoids the
// 419 MB float64 temporary that np.mean([...]) materialises.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

template <typename T>
struct MinOp {
    static inline T apply(T a, T b) { return a < b ? a : b; }
};

template <typename T>
struct MaxOp {
    static inline T apply(T a, T b) { return a > b ? a : b; }
};

// scipy mode='reflect': (d c b a | a b c d | d c b a), symmetric about the
// outer edge of the boundary sample. Must tolerate indices many periods out,
// because a size-10 kernel on a length-4 channel axis reflects repeatedly.
inline int64_t reflect_idx(int64_t j, int64_t n) {
    if (n == 1) return 0;
    const int64_t period = 2 * n;
    j %= period;
    if (j < 0) j += period;
    if (j >= n) j = period - 1 - j;
    return j;
}

// One separable pass along `axis`, van Herk/Gil-Werman.
// Window [lo, hi] (inclusive, hi - lo + 1 == k) is given explicitly rather than
// as a scipy `origin`: the erosion/dilation mirroring is the easiest thing to
// invert by accident, so the caller states the window outright.
template <typename T, typename Op>
void filter_axis(const T* in, T* out, const std::vector<int64_t>& shape, int axis,
                 int lo, int hi, bool parallel, int num_threads) {
    const int ndim = static_cast<int>(shape.size());
    const int64_t n = shape[axis];
    const int k = hi - lo + 1;

    int64_t inner = 1;
    for (int i = axis + 1; i < ndim; ++i) inner *= shape[i];
    int64_t outer = 1;
    for (int i = 0; i < axis; ++i) outer *= shape[i];

    const int64_t pad_left = std::max(0, -lo);
    const int64_t pad_right = std::max(0, hi);
    const int64_t m = n + pad_left + pad_right;
    const int64_t nlines = outer * inner;

#ifdef _OPENMP
#pragma omp parallel if (parallel) num_threads(num_threads)
#endif
    {
        std::vector<T> A(m), g(m), h(m);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int64_t L = 0; L < nlines; ++L) {
            const int64_t o = L / inner;
            const int64_t s = L % inner;
            const T* base_in = in + o * n * inner + s;
            T* base_out = out + o * n * inner + s;

            // Gather the line into a reflected, padded buffer so the hot loop
            // below carries no boundary logic.
            for (int64_t t = 0; t < pad_left; ++t)
                A[t] = base_in[reflect_idx(t - pad_left, n) * inner];
            for (int64_t t = 0; t < n; ++t)
                A[pad_left + t] = base_in[t * inner];
            for (int64_t t = pad_left + n; t < m; ++t)
                A[t] = base_in[reflect_idx(t - pad_left, n) * inner];

            // van Herk: per block of k, a forward running reduce (g) and a
            // backward running reduce (h).
            for (int64_t b = 0; b < m; b += k) {
                const int64_t e = std::min<int64_t>(b + k, m);
                g[b] = A[b];
                for (int64_t t = b + 1; t < e; ++t) g[t] = Op::apply(g[t - 1], A[t]);
                h[e - 1] = A[e - 1];
                for (int64_t t = e - 2; t >= b; --t) h[t] = Op::apply(h[t + 1], A[t]);
            }

            // Any width-k window is then just two lookups.
            for (int64_t i = 0; i < n; ++i) {
                const int64_t sIdx = i + pad_left + lo;
                base_out[i * inner] = Op::apply(h[sIdx], g[sIdx + k - 1]);
            }
        }
    }
}

// out = in - opening(in), where opening = dilate(erode(in)).
// t1/t2 are scratch buffers of npix elements each.
template <typename T>
void tophat_core(const T* in, T* out, const std::vector<int64_t>& shape, int k,
                 T* t1, T* t2, bool parallel, int num_threads) {
    const int ndim = static_cast<int>(shape.size());

    // Dilation's structuring element is the mirror of erosion's, so its window
    // is erosion's reflected through 0. Identical only for odd k.
    const int lo_ero = -(k / 2), hi_ero = k - k / 2 - 1;
    const int lo_dil = -hi_ero, hi_dil = -lo_ero;

    const T* src = in;
    T* dst = t1;
    T* other = t2;

    for (int ax = 0; ax < ndim; ++ax) {
        filter_axis<T, MinOp<T>>(src, dst, shape, ax, lo_ero, hi_ero, parallel, num_threads);
        src = dst;
        std::swap(dst, other);
    }
    for (int ax = 0; ax < ndim; ++ax) {
        filter_axis<T, MaxOp<T>>(src, dst, shape, ax, lo_dil, hi_dil, parallel, num_threads);
        src = dst;
        std::swap(dst, other);
    }

    int64_t npix = 1;
    for (int i = 0; i < ndim; ++i) npix *= shape[i];

    // opening is anti-extensive (opening <= in), so this never wraps for
    // unsigned T — matching scipy, which also subtracts in the input dtype.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel) num_threads(num_threads)
#endif
    for (int64_t i = 0; i < npix; ++i) out[i] = static_cast<T>(in[i] - src[i]);
}

int resolve_threads(int num_threads) {
#ifdef _OPENMP
    if (num_threads <= 0) return omp_get_max_threads();
    return num_threads;
#else
    (void)num_threads;
    return 1;
#endif
}

template <typename T>
py::array tophat_impl(py::array arr, int feature_size, int num_threads) {
    auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(arr);
    if (!a) throw std::runtime_error("could not convert input to a C-contiguous array");

    std::vector<int64_t> shape(a.ndim());
    int64_t npix = 1;
    for (int i = 0; i < a.ndim(); ++i) {
        shape[i] = a.shape(i);
        npix *= shape[i];
    }

    py::array_t<T> out(a.request().shape);
    const T* in_ptr = a.data();
    T* out_ptr = out.mutable_data();
    const int nt = resolve_threads(num_threads);

    {
        py::gil_scoped_release release;
        std::vector<T> t1(npix), t2(npix);
        tophat_core<T>(in_ptr, out_ptr, shape, feature_size, t1.data(), t2.data(), true, nt);
    }
    return out;
}

template <typename T>
py::array batch_mean_impl(py::array arr, int feature_size, int num_threads) {
    auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(arr);
    if (!a) throw std::runtime_error("could not convert input to a C-contiguous array");
    if (a.ndim() < 2) throw std::runtime_error("batch input must have ndim >= 2");

    const int64_t nimg = a.shape(0);
    if (nimg == 0) throw std::runtime_error("batch is empty");

    std::vector<int64_t> shape(a.ndim() - 1);
    std::vector<py::ssize_t> out_shape;
    int64_t npix = 1;
    for (int i = 1; i < a.ndim(); ++i) {
        shape[i - 1] = a.shape(i);
        npix *= a.shape(i);
        out_shape.push_back(a.shape(i));
    }

    py::array_t<double> out(out_shape);
    const T* in_ptr = a.data();
    double* out_ptr = out.mutable_data();
    const int nt = resolve_threads(num_threads);

    {
        py::gil_scoped_release release;
        std::vector<uint64_t> total(npix, 0);

#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
#endif
        {
            // Per-thread accumulator, reduced once at the end: no atomics in
            // the inner loop and no stacked float64 temporary.
            std::vector<uint64_t> acc(npix, 0);
            std::vector<T> t1(npix), t2(npix), res(npix);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
            for (int64_t i = 0; i < nimg; ++i) {
                // Images run parallel to each other, so each top-hat is
                // single-threaded here.
                tophat_core<T>(in_ptr + i * npix, res.data(), shape, feature_size,
                               t1.data(), t2.data(), false, 1);
                for (int64_t j = 0; j < npix; ++j)
                    acc[j] += static_cast<uint64_t>(res[j]);
            }

#ifdef _OPENMP
#pragma omp critical
#endif
            {
                for (int64_t j = 0; j < npix; ++j) total[j] += acc[j];
            }
        }

        // Exact integer sum -> float64 -> divide reproduces np.mean bit-for-bit:
        // every partial sum is an integer well under 2^53, so np.mean's pairwise
        // float64 summation is exact too and the order cannot matter.
        const double denom = static_cast<double>(nimg);
        for (int64_t j = 0; j < npix; ++j)
            out_ptr[j] = static_cast<double>(total[j]) / denom;
    }
    return out;
}

void check_size(int feature_size) {
    if (feature_size < 1) throw std::runtime_error("feature_size must be >= 1");
}

py::array tophat(py::array image, int feature_size, int num_threads) {
    check_size(feature_size);
    const auto dt = image.dtype();
    if (dt.is(py::dtype::of<uint8_t>())) return tophat_impl<uint8_t>(image, feature_size, num_threads);
    if (dt.is(py::dtype::of<uint16_t>())) return tophat_impl<uint16_t>(image, feature_size, num_threads);
    if (dt.is(py::dtype::of<float>())) return tophat_impl<float>(image, feature_size, num_threads);
    if (dt.is(py::dtype::of<double>())) return tophat_impl<double>(image, feature_size, num_threads);
    throw std::runtime_error("unsupported dtype: expected uint8, uint16, float32 or float64");
}

py::array tophat_batch_mean(py::array images, int feature_size, int num_threads) {
    check_size(feature_size);
    const auto dt = images.dtype();
    if (dt.is(py::dtype::of<uint8_t>())) return batch_mean_impl<uint8_t>(images, feature_size, num_threads);
    if (dt.is(py::dtype::of<uint16_t>())) return batch_mean_impl<uint16_t>(images, feature_size, num_threads);
    if (dt.is(py::dtype::of<float>())) return batch_mean_impl<float>(images, feature_size, num_threads);
    if (dt.is(py::dtype::of<double>())) return batch_mean_impl<double>(images, feature_size, num_threads);
    throw std::runtime_error("unsupported dtype: expected uint8, uint16, float32 or float64");
}

int max_threads() { return resolve_threads(0); }

bool has_openmp() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

}  // namespace

PYBIND11_MODULE(morph_tophat_cpp, m) {
    m.doc() = "Parallel white top-hat, bit-exact with scipy.ndimage.white_tophat(size=...)";

    m.def("tophat", &tophat, py::arg("image"), py::arg("feature_size"),
          py::arg("num_threads") = 0,
          "Drop-in for scipy.ndimage.white_tophat(image, size=feature_size).\n"
          "A scalar feature_size applies to every axis, matching scipy.");

    m.def("tophat_batch_mean", &tophat_batch_mean, py::arg("images"),
          py::arg("feature_size"), py::arg("num_threads") = 0,
          "Fused equivalent of np.mean([white_tophat(im, size=f) for im in images], axis=0).\n"
          "images is a stacked array with the batch on axis 0. Returns float64.");

    m.def("max_threads", &max_threads, "OpenMP threads this module will use by default.");
    m.def("has_openmp", &has_openmp, "True if the module was built with OpenMP enabled.");
}
