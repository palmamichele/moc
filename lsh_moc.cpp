// sketch code, adapt this to fmca library
#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads(){ return 1; }
inline int omp_get_thread_num(){ return 0; }
#endif

#include <Eigen/Core>
#include <Eigen/Dense>

extern "C" {
typedef double c_double;
typedef float  c_float;
typedef std::size_t c_size;
}

using Size = std::size_t;
using Index = Eigen::Index;
using Scalar = double;
using MatrixR = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorR = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// small utility: stable 64-bit mix
static inline uint64_t mix64(uint64_t x) {
    x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
    x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
    x = x ^ (x >> 33);
    return x;
}

// deterministic order-sensitive hash for a vector of 64-bit ints
static inline uint64_t hash_key(const int64_t* keys, Size K) {
    uint64_t h = 1469598103934665603ULL; // FNV offset basis
    for (Size i = 0; i < K; ++i) {
        uint64_t v = static_cast<uint64_t>(keys[i]);
        h ^= mix64(v + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2));
        h *= 1099511628211ULL; // FNV prime
    }
    return h;
}

// compute Euclidean distance between two rows
static inline float euclidean_distance_row(const double* a, const double* b, Size dim) {
    double s = 0.0;
    for (Size k = 0; k < dim; ++k) {
        double d = a[k] - b[k];
        s += d * d;
    }
    return static_cast<float>(std::sqrt(s));
}

static std::vector<float> compute_lsh_moc_eigen(const c_double* x_data,
                                                Size n,
                                                Size dx_dim,
                                                const c_double* f_data,
                                                Size f_dim,
                                                const c_double* a_data, // L * K * dx_dim
                                                const c_double* b_data, // L * K
                                                Size K,
                                                Size L,
                                                double w,   // bucket width for LSH (same as self.w)
                                                double h_binwidth,
                                                Size block = 128)
{
    if (!x_data || !f_data || !a_data || !b_data) throw std::invalid_argument("null pointer");
    if (n == 0) return {};
    if (dx_dim == 0 || f_dim == 0) throw std::invalid_argument("invalid dims");
    if (K == 0 || L == 0) throw std::invalid_argument("K and L must be > 0");
    if (w <= 0.0) throw std::invalid_argument("w must be > 0");
    if (h_binwidth <= 0.0) throw std::invalid_argument("h_binwidth > 0");

    // Map raw arrays to Eigen maps (row-major)
    Eigen::Map<const MatrixR> X(x_data, static_cast<Index>(n), static_cast<Index>(dx_dim));
    Eigen::Map<const MatrixR> F(f_data, static_cast<Index>(n), static_cast<Index>(f_dim));

    // Determine maximum spatial bin we might produce by sampling pairwise distances in cheap way:
    // To keep implementation simple, we will estimate a reasonable maximum bin by scanning a few pairwise blocks;
    // but to be robust, we will compute bin index on the fly and grow per-thread vectors as needed.

    int nthreads = std::max(1, omp_get_max_threads());
    std::vector<std::vector<double>> localBs(nthreads); // each will be grown to necessary bins

    // Precompute pointers to row starts for faster C access
    std::vector<const double*> x_row_ptrs(n);
    std::vector<const double*> f_row_ptrs(n);
    for (Size i = 0; i < n; ++i) {
        x_row_ptrs[i] = x_data + i * dx_dim;
        f_row_ptrs[i] = f_data + i * f_dim;
    }

    // For hash key building
    std::vector<int64_t> keyvec(K);

    // Process each table independently (parallel over tables)
    #pragma omp parallel for schedule(dynamic)
    for (Size ell = 0; ell < L; ++ell) {
        int tid = omp_get_thread_num();
        auto &localB = localBs[tid];

        // offsets into a_data/b_data
        const c_double* a_table = a_data + (Size)ell * K * dx_dim;
        const c_double* b_table = b_data + (Size)ell * K;

        // 1) compute integer keys for all points, store to vector<int64_t> per point or compute on the fly
        // We'll compute int keys per-point into contiguous buffer: keys_per_point[n*K]
        std::vector<int64_t> keys_per_point;
        keys_per_point.resize((Size)n * K);

        for (Size i = 0; i < n; ++i) {
            const double* xi = x_row_ptrs[i];
            for (Size kk = 0; kk < K; ++kk) {
                // a_table block for kk: a_table + kk*dx_dim
                const double* a_vec = a_table + kk * dx_dim;
                double dot = 0.0;
                for (Size d = 0; d < dx_dim; ++d) dot += a_vec[d] * xi[d];
                double val = (dot + b_table[kk]) / w;
                long long iv = static_cast<long long>(std::floor(val));
                keys_per_point[i * K + kk] = static_cast<int64_t>(iv);
            }
        }

        // 2) bucket points by hashed key
        std::unordered_map<uint64_t, std::vector<int>> buckets;
        buckets.reserve(n * 2 / 3 + 16);

        for (Size i = 0; i < n; ++i) {
            const int64_t* kptr = &keys_per_point[i * K];
            uint64_t h = hash_key(kptr, K);
            buckets[h].push_back(static_cast<int>(i));
        }

        // 3) for each bucket, examine pairs
        for (auto &kv : buckets) {
            const std::vector<int> &pts = kv.second;
            Size m = pts.size();
            if (m <= 1) continue;

            // pairwise O(m^2) — if m is huge this could be heavy; hope L and K keep buckets small.
            for (Size ii = 0; ii < m; ++ii) {
                int iidx = pts[ii];
                const double* xi = x_row_ptrs[iidx];
                const double* fi = f_row_ptrs[iidx];
                for (Size jj = ii + 1; jj < m; ++jj) {
                    int jidx = pts[jj];
                    const double* xj = x_row_ptrs[jidx];
                    const double* fj = f_row_ptrs[jidx];

                    // spatial distance
                    float sij = euclidean_distance_row(xi, xj, dx_dim);
                    int bin = static_cast<int>(std::floor(sij / h_binwidth));
                    if (bin < 0) bin = 0;
                    Size bidx = static_cast<Size>(bin);

                    // compute feature euclidean dist between fi and fj
                    double s = 0.0;
                    for (Size dd = 0; dd < f_dim; ++dd) {
                        double ddiff = fi[dd] - fj[dd];
                        s += ddiff * ddiff;
                    }
                    double dij = std::sqrt(s);

                    // ensure localB is big enough (resize if needed)
                    if (localB.size() <= bidx) localB.resize(bidx + 1, 0.0);
                    if (localB[bidx] < dij) localB[bidx] = dij;
                }
            }
        }
    } // end parallel over tables

    // Merge localBs into global B (double)
    std::size_t maxbins = 0;
    for (auto &v : localBs) if (v.size() > maxbins) maxbins = v.size();
    std::vector<double> B(maxbins, 0.0);

    for (int t = 0; t < nthreads; ++t) {
        auto &v = localBs[t];
        for (Size b = 0; b < v.size(); ++b) {
            if (B[b] < v[b]) B[b] = v[b];
        }
    }

    // compute prefix maxima and convert to float
    std::vector<float> Bprefix;
    if (!B.empty()) {
        Bprefix.reserve(B.size());
        double runmax = B[0];
        Bprefix.push_back(static_cast<float>(runmax));
        for (Size i = 1; i < B.size(); ++i) {
            if (B[i] > runmax) runmax = B[i];
            Bprefix.push_back(static_cast<float>(runmax));
        }
    }
    return Bprefix;
}

extern "C" {

c_float* compute_lsh_moc_c(const c_double* x_data,
                           c_size n,
                           c_size dx_dim,
                           const c_double* f_data,
                           c_size f_dim,
                           const c_double* a_data,
                           const c_double* b_data,
                           c_size K,
                           c_size L,
                           double w,
                           double h,
                           c_size* out_len)
{
    try {
        if (!x_data || !f_data || !a_data || !b_data || !out_len) return nullptr;
        if (n == 0) return nullptr;
        if (dx_dim == 0 || f_dim == 0) return nullptr;
        if (K == 0 || L == 0) return nullptr;
        if (w <= 0.0 || h <= 0.0) return nullptr;

        auto B = compute_lsh_moc_eigen(x_data, n, dx_dim, f_data, f_dim, a_data, b_data, K, L, w, h);

        c_size len = B.size();
        if (len == 0) {
            *out_len = 0;
            return nullptr;
        }
        c_float* out = static_cast<c_float*>(std::malloc(len * sizeof(c_float)));
        if (!out) { *out_len = 0; return nullptr; }
        for (c_size i = 0; i < len; ++i) out[i] = B[i];
        *out_len = len;
        return out;

    } catch (const std::exception& e) {
        if (out_len) *out_len = 0;
        return nullptr;
    }
}

void free_buffer_lsh(c_float* buf) {
    if (buf) std::free(buf);
}

} // extern "C"
