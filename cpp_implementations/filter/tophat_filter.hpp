// tophat_filter.hpp -- dependency-free C++ white-tophat high-pass filter for
// tweezer-array images. This is the *filter only* extract of the fast_detection
// C++ port (trap detection / labelling deliberately omitted): all this folder
// does is the high-pass that runs on every frame before readout.
//
// Design goals:
//   * Bit-for-bit match with scipy.ndimage.white_tophat(img, size=N,
//     mode='reflect'). Morphology is min/max + subtraction, which involve no
//     floating-point rounding, so an exact match is achievable.
//   * Be genuinely faster than scipy's Cython by using an O(n) sliding-window
//     min/max (van Herk / Gil-Werman, monotonic prefix/suffix) instead of
//     scipy's O(n * window) approach -- so the win does not shrink with size.
//
// Header-only and templated on the pixel type (float / double).

#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>

namespace tophat {

// --------------------------------------------------------------------------- //
// scipy 'reflect' boundary: sample(-1) == sample(0), period 2n.
//   ... d c b a | a b c d | d c b a ...
// --------------------------------------------------------------------------- //
inline int reflect_index(int p, int n) {
    if (n == 1) return 0;
    int m = 2 * n;
    p %= m;
    if (p < 0) p += m;
    if (p >= n) p = m - 1 - p;
    return p;
}

// Scratch buffers reused across every line so the hot path never allocates.
template <typename T>
struct Scratch {
    std::vector<T> pad, g, h, col, ocol, tmp;
    void ensure(int height, int width, int maxpad) {
        int maxdim = std::max(height, width);
        pad.resize(maxdim + maxpad);
        g.resize(maxdim + maxpad);
        h.resize(maxdim + maxpad);
        col.resize(maxdim);
        ocol.resize(maxdim);
        tmp.resize(static_cast<size_t>(height) * width);
    }
};

// 1-D sliding-window extremum, window offsets [lo, hi] (inclusive) relative to
// the output index, reflect padding. `IsMin` -> erosion (min) / dilation (max).
//
// Van Herk / Gil-Werman algorithm: O(n) with ~3 comparisons per element,
// independent of the window size, and allocation-free (uses caller scratch).
template <typename T, bool IsMin>
inline void line_extremum(const T* in, T* out, int n, int lo, int hi, Scratch<T>& s) {
    const int left = -lo;             // left padding
    const int k = hi - lo + 1;        // window width
    const int padn = n + left + hi;   // padded length; outputs align at [i, i+k-1]

    T* pad = s.pad.data();
    T* g = s.g.data();
    T* h = s.h.data();

    for (int j = 0; j < padn; ++j)
        pad[j] = in[reflect_index(j - left, n)];

    auto better = [](T a, T b) { return IsMin ? (a < b ? a : b) : (a > b ? a : b); };

    // Block-wise prefix (g) and suffix (h) extrema over blocks of width k.
    for (int s0 = 0; s0 < padn; s0 += k) {
        int end = std::min(s0 + k, padn);
        g[s0] = pad[s0];
        for (int j = s0 + 1; j < end; ++j) g[j] = better(g[j - 1], pad[j]);
        h[end - 1] = pad[end - 1];
        for (int j = end - 2; j >= s0; --j) h[j] = better(h[j + 1], pad[j]);
    }
    for (int i = 0; i < n; ++i) {
        int r = i + k - 1;
        out[i] = (r < padn) ? better(h[i], g[r]) : h[i];
    }
}

// Separable 2-D extremum filter (rows then columns), reflect border.
template <typename T, bool IsMin>
void separable_extremum(const T* in, T* out, int h, int w, int lo, int hi, Scratch<T>& s) {
    T* tmp = s.tmp.data();
    // Along rows (axis=1).
    for (int r = 0; r < h; ++r)
        line_extremum<T, IsMin>(in + static_cast<size_t>(r) * w,
                                tmp + static_cast<size_t>(r) * w, w, lo, hi, s);
    // Along columns (axis=0).
    T* col = s.col.data();
    T* ocol = s.ocol.data();
    for (int c = 0; c < w; ++c) {
        for (int r = 0; r < h; ++r) col[r] = tmp[static_cast<size_t>(r) * w + c];
        line_extremum<T, IsMin>(col, ocol, h, lo, hi, s);
        for (int r = 0; r < h; ++r) out[static_cast<size_t>(r) * w + c] = ocol[r];
    }
}

// White tophat = image - opening, opening = dilation(erosion).
// For scipy size=N (flat square SE): erosion window offsets [-N/2, N-1-N/2];
// dilation mirrors them to [-(N-1-N/2), N/2].
template <typename T>
void white_tophat(const T* in, T* out, int h, int w, int size,
                  Scratch<T>& s, std::vector<T>& eroded, std::vector<T>& opened) {
    const int er_lo = -(size / 2);
    const int er_hi = size - 1 - (size / 2);
    const size_t npx = static_cast<size_t>(h) * w;
    eroded.resize(npx);
    opened.resize(npx);

    separable_extremum<T, true>(in, eroded.data(), h, w, er_lo, er_hi, s);
    // Dilation offsets are the mirror of the erosion offsets.
    separable_extremum<T, false>(eroded.data(), opened.data(), h, w, -er_hi, -er_lo, s);
    for (size_t i = 0; i < npx; ++i) out[i] = in[i] - opened[i];
}

// Convenience single-image entry point (allocates its own scratch).
template <typename T>
void white_tophat(const T* in, T* out, int h, int w, int size) {
    Scratch<T> s;
    s.ensure(h, w, 2 * size);
    std::vector<T> eroded, opened;
    white_tophat<T>(in, out, h, w, size, s, eroded, opened);
}

// Batch white tophat over `n` frames, reusing all scratch across frames.
template <typename T>
void batch_white_tophat(const T* in, T* out, int n, int h, int w, int size) {
    Scratch<T> s;
    s.ensure(h, w, 2 * size);
    std::vector<T> eroded, opened;
    const size_t fr = static_cast<size_t>(h) * w;
    for (int i = 0; i < n; ++i)
        white_tophat<T>(in + i * fr, out + i * fr, h, w, size, s, eroded, opened);
}

}  // namespace tophat
