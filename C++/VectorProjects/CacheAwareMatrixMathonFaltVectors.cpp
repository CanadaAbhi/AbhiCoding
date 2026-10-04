// vec_matmul.cpp (g++ -O3 -march=native -std=c++17 vec_matmul.cpp)
#include <vector>
#include <chrono>
#include <cstdio>
#include <random>

struct Matrix {
    std::vector<float> d;        // row-major flat storage: ONE allocation
    int n;
    explicit Matrix(int n_) : d((size_t)n_ * n_, 0.f), n(n_) {}
    float&       at(int r, int c)       { return d[(size_t)r * n + c]; }
    const float& at(int r, int c) const { return d[(size_t)r * n + c]; }
};

void matmul_naive(const Matrix& A, const Matrix& B, Matrix& C) {
    int n = A.n;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float acc = 0.f;
            for (int k = 0; k < n; ++k)
                acc += A.at(i, k) * B.at(k, j);   // B strided: cache-hostile
            C.at(i, j) = acc;
        }
}

void matmul_ikj(const Matrix& A, const Matrix& B, Matrix& C) {
    int n = A.n;                                   // loop-reorder: B row-wise
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            float a = A.at(i, k);
            for (int j = 0; j < n; ++j)
                C.at(i, j) += a * B.at(k, j);      // unit-stride, vectorizes
        }
}

void matmul_blocked(const Matrix& A, const Matrix& B, Matrix& C, int T = 64) {
    int n = A.n;
    for (int ii = 0; ii < n; ii += T)
        for (int kk = 0; kk < n; kk += T)
            for (int jj = 0; jj < n; jj += T)
                for (int i = ii; i < std::min(ii + T, n); ++i)
                    for (int k = kk; k < std::min(kk + T, n); ++k) {
                        float a = A.at(i, k);
                        for (int j = jj; j < std::min(jj + T, n); ++j)
                            C.at(i, j) += a * B.at(k, j);
                    }
}

template <typename F>
double bench(F f, const char* name, const Matrix& A, const Matrix& B, int n) {
    Matrix C(n);
    auto t0 = std::chrono::steady_clock::now();
    f(A, B, C);
    auto ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
    double gflops = 2.0 * n * n * (double)n / (ms * 1e6);
    std::printf("%-16s %8.1f ms  %6.2f GFLOP/s\n", name, ms, gflops);
    return ms;
}

int main() {
    const int n = 1024;
    Matrix A(n), B(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1, 1);
    for (auto& x : A.d) x = u(rng);
    for (auto& x : B.d) x = u(rng);

    bench(matmul_naive,   "naive ijk", A, B, n);
    bench(matmul_ikj,     "reordered ikj", A, B, n);
    bench([](auto& a, auto& b, auto& c){ matmul_blocked(a, b, c); },
          "blocked 64", A, B, n);
}
