// bprefix_lib_openmp.cpp
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cstddef>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads(){ return 1; }
inline int omp_get_thread_num(){ return 0; }
#endif

extern "C" {
typedef double c_double;
typedef float  c_float;
typedef std::size_t c_size;
}

using Point    = std::vector<double>;
using Dataset  = std::vector<Point>;
using FDataset = std::vector<Dataset>;

float euclideanDistance(const Point& a, const Point& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Dimension mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return static_cast<float>(std::sqrt(sum));
}

// Helper that computes max bin index (one pass)
static int compute_max_binindex(const Dataset& x, double h) {
    const size_t n = x.size();
    long maxBin = 0;
    #pragma omp parallel
    {
        long local_max = 0;
        #pragma omp for nowait schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sij = euclideanDistance(x[i], x[j]);
                int binindex = static_cast<int>(std::floor(sij / h));
                if (binindex < 0) binindex = 0;
                if (binindex > local_max) local_max = binindex;
            }
        }
        #pragma omp critical
        { if (local_max > maxBin) maxBin = local_max; }
    }
    return static_cast<int>(maxBin);
}

std::vector<float> computeBprefix_openmp(const Dataset& x,
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

    // 1) find maximum bin index (so we can allocate fixed-size arrays)
    int maxBin = compute_max_binindex(x, h);
    const size_t bins = static_cast<size_t>(maxBin) + 1;

    // Global B initialized to 0
    std::vector<float> B(bins, 0.0f);

    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;

    // thread-local buffers: one vector per thread
    std::vector<std::vector<float>> localBs(nthreads, std::vector<float>(bins, 0.0f));

    // 2) parallel fill localB per thread
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto &localB = localBs[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sij = dx(x[i], x[j]);
                float dij = dy(f[idxf][i], f[idxf][j]);
                int binindex = static_cast<int>(std::floor(sij / h));
                if (binindex < 0) binindex = 0;
                // safe: each thread updates its own localB
                if (localB[static_cast<size_t>(binindex)] < dij)
                    localB[static_cast<size_t>(binindex)] = dij;
            }
        }
    }

    // 3) merge localBs into global B using thread-safe max (single thread will do it)
    for (int t = 0; t < nthreads; ++t) {
        for (size_t b = 0; b < bins; ++b) {
            if (B[b] < localBs[t][b]) B[b] = localBs[t][b];
        }
    }

    // 4) prefix max
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

// C ABI wrapper 
extern "C" {

c_float* compute_bprefix_c(const c_double* x_data,
                           c_size n,
                           c_size dx_dim,
                           const c_double* f_data,
                           c_size f_dim,
                           c_size idxf,
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

        // Reconstruct f as FDataset with a single feature-set f[0]
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

       

        auto Bprefix = computeBprefix_openmp(x, f, 0, h, euclideanDistance, euclideanDistance);

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
        *out_len = 0;
        return nullptr;
    }
}

void free_buffer(c_float* buf) {
    if (buf) std::free(buf);
}

} // extern "C"
#include <cstdio>

extern "C" {
void omp_debug_print() {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        #pragma omp critical
        {
            printf("OMP_DEBUG: tid=%d / nthreads=%d\n", tid, nthreads);
            fflush(stdout);
        }
    }
}
}
