// sketch code, adapt this to fmca library
#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <iostream>

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

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>


using Scalar = double;
using Index  = Eigen::Index;
using Size   = std::size_t;
using MatrixR = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorR = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// block shall fit comfortably in L2 cache
static const Size DEFAULT_BLOCK = 128;




static inline float euclideanDistance_vec(const double* a, const double* b, Size dim) {
    double s = 0.0;
    for (Size k = 0; k < dim; ++k) {
        double d = a[k] - b[k];
        s += d * d;
    }
    return static_cast<float>(std::sqrt(s));
}

static std::vector<float> computeBprefix_eigen(const c_double* x_data,
                                               Size n,
                                               Size dx_dim,
                                               const c_double* f_data,
                                               Size f_dim,
                                               Size idxf_unused, 
                                               double h,
                                               Size block = DEFAULT_BLOCK)
{
    if (!x_data || !f_data) throw std::invalid_argument("null data pointer");
    if (n == 0) return {};
    if (dx_dim == 0 || f_dim == 0) throw std::invalid_argument("invalid dimension");
    if (h <= 0.0) throw std::invalid_argument("h must be > 0");

    // Map raw C arrays (row-major: n rows, dx_dim columns)
    Eigen::Map<const MatrixR> X(x_data, static_cast<Index>(n), static_cast<Index>(dx_dim));
    Eigen::Map<const MatrixR> F(f_data, static_cast<Index>(n), static_cast<Index>(f_dim));

    // Precompute squared norms for each row 
    VectorR Xnorm2(n), Fnorm2(n);
    #pragma omp parallel for schedule(static)
    for (Size i = 0; i < n; ++i) {
        Xnorm2(static_cast<Index>(i)) = X.row(static_cast<Index>(i)).squaredNorm();
        Fnorm2(static_cast<Index>(i)) = F.row(static_cast<Index>(i)).squaredNorm();
    }

    // 1) Determine maximum bin index (one blocked pass).
    long global_max_bin = 0;
    Size nthreads = static_cast<Size>(std::max(1, omp_get_max_threads()));

    #pragma omp parallel
    {
        long local_max = 0;
        #pragma omp for schedule(static)
        for (Size i0 = 0; i0 < n; i0 += block) {
            Size i1 = std::min(n, i0 + block);
            for (Size j0 = 0; j0 < n; j0 += block) {
                Size j1 = std::min(n, j0 + block);

                // Create Eigen block views (no copy)
                auto A = X.block(static_cast<Index>(i0), 0, static_cast<Index>(i1 - i0), static_cast<Index>(dx_dim));
                auto B = X.block(static_cast<Index>(j0), 0, static_cast<Index>(j1 - j0), static_cast<Index>(dx_dim));

                // an and bn are temporary small vectors (block sized)
                Eigen::VectorXd an = A.rowwise().squaredNorm();
                Eigen::VectorXd bn = B.rowwise().squaredNorm();

                // cross = A * B^T is expressed lazily; assign to matrix to evaluate.
                Eigen::MatrixXd cross = A * B.transpose();

                // compute distances and update local max bin
                for (int ii = 0; ii < cross.rows(); ++ii) {
                    for (int jj = 0; jj < cross.cols(); ++jj) {
                        double sqd = an[ii] + bn[jj] - 2.0 * cross(ii, jj);
                        if (sqd < 0.0 && sqd > -1e-12) sqd = 0.0;
                        double sij = std::sqrt(sqd);
                        int bin = static_cast<int>(std::floor(sij / h));
                        if (bin < 0) bin = 0;
                        if (bin > local_max) local_max = bin;
                    }
                }
            }
        }
        #pragma omp critical
        if (local_max > global_max_bin) global_max_bin = local_max;
    } // end parallel

    if (global_max_bin < 0) global_max_bin = 0;
    Size bins = static_cast<Size>(global_max_bin) + 1;

    // 2) Thread-local maxima arrays (double precision for accumulation)
    std::vector<std::vector<double>> localBs(nthreads, std::vector<double>(bins, 0.0));

    // 3) Blocked computation: for each block compute feature distances and spatial bins in a fused expression
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto &localB = localBs[static_cast<Size>(tid)];

        #pragma omp for schedule(static)
        for (Size i0 = 0; i0 < n; i0 += block) {
            Size i1 = std::min(n, i0 + block);
            for (Size j0 = 0; j0 < n; j0 += block) {
                Size j1 = std::min(n, j0 + block);

                
                auto XA = X.block(static_cast<Index>(i0), 0, static_cast<Index>(i1 - i0), static_cast<Index>(dx_dim));
                auto XB = X.block(static_cast<Index>(j0), 0, static_cast<Index>(j1 - j0), static_cast<Index>(dx_dim));
                auto FA = F.block(static_cast<Index>(i0), 0, static_cast<Index>(i1 - i0), static_cast<Index>(f_dim));
                auto FB = F.block(static_cast<Index>(j0), 0, static_cast<Index>(j1 - j0), static_cast<Index>(f_dim));

                // Block norms (small vectors)
                Eigen::VectorXd an = XA.rowwise().squaredNorm();
                Eigen::VectorXd bn = XB.rowwise().squaredNorm();
                Eigen::VectorXd af = FA.rowwise().squaredNorm();
                Eigen::VectorXd bf = FB.rowwise().squaredNorm();

           
                Eigen::MatrixXd crossX = XA * XB.transpose(); // spatial cross
                Eigen::MatrixXd crossF = FA * FB.transpose(); // feature cross

                // Now iterate over small block to compute sij and dij, update per-bin maxima.
                for (int ii = 0; ii < crossF.rows(); ++ii) {
                    for (int jj = 0; jj < crossF.cols(); ++jj) {
                        double sqdF = af[ii] + bf[jj] - 2.0 * crossF(ii, jj);
                        if (sqdF < 0.0 && sqdF > -1e-12) sqdF = 0.0;
                        double dij = std::sqrt(sqdF);

                        double sqdX = an[ii] + bn[jj] - 2.0 * crossX(ii, jj);
                        if (sqdX < 0.0 && sqdX > -1e-12) sqdX = 0.0;
                        double sij = std::sqrt(sqdX);

                        int binindex = static_cast<int>(std::floor(sij / h));
                        if (binindex < 0) binindex = 0;
                        Size bidx = static_cast<Size>(binindex);
                        if (bidx >= bins) bidx = bins - 1; // safety; shouldn't happen
                        if (localB[bidx] < dij) localB[bidx] = dij;
                    }
                }
            }
        }
    } // end parallel

    // 4) Merge localBs into global B
    std::vector<double> B(bins, 0.0);
    for (Size t = 0; t < nthreads; ++t) {
        for (Size b = 0; b < bins; ++b) {
            if (B[b] < localBs[t][b]) B[b] = localBs[t][b];
        }
    }

    // 5) compute prefix maxima and return as floats
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

        // Call Eigen-based core (no copying into std::vector)
        auto Bprefix = computeBprefix_eigen(x_data, n, dx_dim, f_data, f_dim, idxf, h);

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
        if (out_len) *out_len = 0;
        return nullptr;
    }
}

void free_buffer(c_float* buf) {
    if (buf) std::free(buf);
}

} // extern "C"



// Compute maximum Euclidean distance between all points in X
static double computeMaxDistance_eigen(const c_double* x_data, Size n, Size dx_dim, Size block = DEFAULT_BLOCK) {
    if (!x_data || n == 0 || dx_dim == 0) return 0.0;

    Eigen::Map<const MatrixR> X(x_data, static_cast<Index>(n), static_cast<Index>(dx_dim));
    VectorR Xnorm2(n);
    #pragma omp parallel for
    for (Size i = 0; i < n; ++i) Xnorm2(static_cast<Index>(i)) = X.row(static_cast<Index>(i)).squaredNorm();

    double global_max = 0.0;

    #pragma omp parallel
    {
        double local_max = 0.0;
        #pragma omp for schedule(static)
        for (Size i0 = 0; i0 < n; i0 += block) {
            Size i1 = std::min(n, i0 + block);
            for (Size j0 = i0; j0 < n; j0 += block) { // symmetry, start at i0
                Size j1 = std::min(n, j0 + block);

                auto A = X.block(static_cast<Index>(i0), 0, static_cast<Index>(i1 - i0), static_cast<Index>(dx_dim));
                auto B = X.block(static_cast<Index>(j0), 0, static_cast<Index>(j1 - j0), static_cast<Index>(dx_dim));

                Eigen::VectorXd an = A.rowwise().squaredNorm();
                Eigen::VectorXd bn = B.rowwise().squaredNorm();
                Eigen::MatrixXd cross = A * B.transpose();

                for (int ii = 0; ii < cross.rows(); ++ii) {
                    for (int jj = 0; jj < cross.cols(); ++jj) {
                        if (i0 + ii == j0 + jj) continue; // skip diagonal
                        double sqd = an[ii] + bn[jj] - 2.0 * cross(ii, jj);
                        if (sqd < 0.0 && sqd > -1e-12) sqd = 0.0;
                        double d = std::sqrt(sqd);
                        if (d > local_max) local_max = d;
                    }
                }
            }
        }
        #pragma omp critical
        if (local_max > global_max) global_max = local_max;
    }
    return global_max;
}

// Compute exact Lipschitz constant: max ||f_i - f_j|| / ||x_i - x_j||
static double computeLipschitz_eigen(const c_double* x_data,
                                     const c_double* f_data,
                                     Size n,
                                     Size dx_dim,
                                     Size f_dim,
                                     Size block = DEFAULT_BLOCK) {
    if (!x_data || !f_data || n < 2 || dx_dim == 0 || f_dim == 0) return 0.0;

    Eigen::Map<const MatrixR> X(x_data, static_cast<Index>(n), static_cast<Index>(dx_dim));
    Eigen::Map<const MatrixR> F(f_data, static_cast<Index>(n), static_cast<Index>(f_dim));

    VectorR Xnorm2(n), Fnorm2(n);
    #pragma omp parallel for
    for (Size i = 0; i < n; ++i) {
        Xnorm2(static_cast<Index>(i)) = X.row(static_cast<Index>(i)).squaredNorm();
        Fnorm2(static_cast<Index>(i)) = F.rowwise().squaredNorm()(i); // squaredNorm of f_i
    }

    double global_max = 0.0;

    #pragma omp parallel
    {
        double local_max = 0.0;
        #pragma omp for schedule(static)
        for (Size i0 = 0; i0 < n; i0 += block) {
            Size i1 = std::min(n, i0 + block);
            for (Size j0 = i0 + 1; j0 < n; j0 += block) { // avoid i==j
                Size j1 = std::min(n, j0 + block);

                auto XA = X.block(static_cast<Index>(i0), 0, static_cast<Index>(i1 - i0), static_cast<Index>(dx_dim));
                auto XB = X.block(static_cast<Index>(j0), 0, static_cast<Index>(j1 - j0), static_cast<Index>(dx_dim));
                auto FA = F.block(static_cast<Index>(i0), 0, static_cast<Index>(i1 - i0), static_cast<Index>(f_dim));
                auto FB = F.block(static_cast<Index>(j0), 0, static_cast<Index>(j1 - j0), static_cast<Index>(f_dim));

                Eigen::VectorXd an = XA.rowwise().squaredNorm();
                Eigen::VectorXd bn = XB.rowwise().squaredNorm();
                Eigen::VectorXd af = FA.rowwise().squaredNorm();
                Eigen::VectorXd bf = FB.rowwise().squaredNorm();

                Eigen::MatrixXd crossX = XA * XB.transpose();
                Eigen::MatrixXd crossF = FA * FB.transpose();

                for (int ii = 0; ii < crossX.rows(); ++ii) {
                    for (int jj = 0; jj < crossX.cols(); ++jj) {
                        double sqdX = an[ii] + bn[jj] - 2.0 * crossX(ii, jj);
                        if (sqdX <= 1e-12) continue; // skip too-close or identical points
                        double dX = std::sqrt(sqdX);

                        double sqdF = af[ii] + bf[jj] - 2.0 * crossF(ii, jj);
                        if (sqdF < 0.0 && sqdF > -1e-12) sqdF = 0.0;
                        double dF = std::sqrt(sqdF);

                        double ratio = dF / dX;
                        if (ratio > local_max) local_max = ratio;
                    }
                }
            }
        }
        #pragma omp critical
        if (local_max > global_max) global_max = local_max;
    }

    return global_max;
}

extern "C" {

// Compute maximum Euclidean distance between all points
c_double compute_max_distance_c(const c_double* x_data,
                                c_size n,
                                c_size dx_dim) {
    try {
        return computeMaxDistance_eigen(x_data, n, dx_dim);
    } catch (...) {
        return -1.0; // error
    }
}

// Compute exact Lipschitz constant L = max ||f_i - f_j|| / ||x_i - x_j||
c_double compute_lipschitz_c(const c_double* x_data,
                              const c_double* f_data,
                              c_size n,
                              c_size dx_dim,
                              c_size f_dim) {
    try {
        return computeLipschitz_eigen(x_data, f_data, n, dx_dim, f_dim);
    } catch (...) {
        return -1.0; // error
    }
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
