// bprefix_lib.cpp
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>   // for malloc/free
#include <cstddef>   // size_t
#include <cstring>   // memcpy

extern "C" {

typedef double c_double;
typedef float  c_float;
typedef std::size_t c_size;


}

using Point    = std::vector<double>;
using Dataset  = std::vector<Point>;
using FDataset = std::vector<Dataset>; // f[k][i]

float euclideanDistance(const Point& a, const Point& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Dimension mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return static_cast<float>(std::sqrt(sum));
}

std::vector<float> computeBprefix(const Dataset& x,
                                  const FDataset& f,
                                  size_t idxf,
                                  double h,
                                  std::function<float(const Point&, const Point&)> dx,
                                  std::function<float(const Point&, const Point&)> dy)
{
    const size_t n = x.size();
    if (n == 0) throw std::runtime_error("empty dataset x");
    if (idxf >= f.size()) throw std::out_of_range("idxf out of range");
    if (f[idxf].size() != n) throw std::runtime_error("f[idxf] size must equal x size");
    if (h <= 0.0) throw std::invalid_argument("h must be > 0");

    std::vector<float> B;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sij = dx(x[i], x[j]);
            float dij = dy(f[idxf][i], f[idxf][j]);

            int binindex = static_cast<int>(std::floor(sij / h));
            if (binindex < 0) binindex = 0;

            if (static_cast<size_t>(binindex) >= B.size()) {
                B.resize(static_cast<size_t>(binindex) + 1, 0.0f);
            }

            B[static_cast<size_t>(binindex)] = std::max(B[static_cast<size_t>(binindex)], dij);
        }
    }

    std::vector<float> Bprefix;
    if (!B.empty()) {
        Bprefix.reserve(B.size());
        float runmax = B[0];
        Bprefix.push_back(runmax);
        for (size_t t = 1; t < B.size(); ++t) {
            runmax = std::max(runmax, B[t]);
            Bprefix.push_back(runmax);
        }
    }

    return Bprefix;
}

extern "C" {

c_float* compute_bprefix_c(const c_double* x_data,
                           c_size n,
                           c_size dx_dim,
                           const c_double* f_data,
                           c_size f_dim,
                           c_size idxf, // kept for API parity; not used here
                           double h,
                           c_size* out_len)
{
    try {
        if (!x_data || !f_data || !out_len) return nullptr;
        if (n == 0) return nullptr;
        if (h <= 0.0) return nullptr;

        // Reconstruct Dataset x from raw array (copy)
        Dataset x;
        x.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            Point p;
            p.reserve(dx_dim);
            const c_double* base = x_data + i * dx_dim;
            for (size_t d = 0; d < dx_dim; ++d) p.push_back(base[d]);
            x.push_back(std::move(p));
        }

        // Reconstruct f as FDataset with single feature-set f[0] (since caller passed f[idxf] flattened)
        FDataset f;
        f.emplace_back();
        f[0].reserve(n);
        for (size_t i = 0; i < n; ++i) {
            Point p;
            p.reserve(f_dim);
            const c_double* base = f_data + i * f_dim;
            for (size_t d = 0; d < f_dim; ++d) p.push_back(base[d]);
            f[0].push_back(std::move(p));
        }

        // compute Bprefix for idxf = 0 (because we passed only one feature-set)
        auto Bprefix = computeBprefix(x, f, 0, h, euclideanDistance, euclideanDistance);

        // allocate C buffer and copy results
        c_size len = Bprefix.size();
        if (len == 0) {
            *out_len = 0;
            return nullptr;
        }

        c_float* out = static_cast<c_float*>(std::malloc(len * sizeof(c_float)));
        if (!out) {
            *out_len = 0;
            return nullptr;
        }
        for (c_size i = 0; i < len; ++i) out[i] = Bprefix[i];

        *out_len = len;
        return out;

    } catch (const std::exception& e) {
        // On error return nullptr and out_len = 0
        *out_len = 0;
        return nullptr;
    }
}

/* free helper */
void free_buffer(c_float* buf) {
    if (buf) std::free(buf);
}

} // extern "C"
