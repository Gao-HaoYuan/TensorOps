#include <cuda_runtime.h>

#include "mini_gtest.h"
#include "device_launch.cuh"
#include "half.h"

using witin::Half;

struct AddOp { template<typename A, typename B> HOST_DEVICE auto operator()(A a, B b) const { return a + b; } };
struct SubOp { template<typename A, typename B> HOST_DEVICE auto operator()(A a, B b) const { return a - b; } };
struct MulOp { template<typename A, typename B> HOST_DEVICE auto operator()(A a, B b) const { return a * b; } };
struct DivOp { template<typename A, typename B> HOST_DEVICE auto operator()(A a, B b) const { return a / b; } };

/// -------------------------- Host Test -------------------------------
template <typename T1, typename T2>
void TestArithmeticOps(T1 a, T2 b, float tol = 1e-3f) {
    using std::is_same;
    using std::is_floating_point;

    auto to_float = [](auto x) -> float { return static_cast<float>(x); };

    EXPECT_NEAR(to_float(a + b), to_float(a) + to_float(b), tol);
    EXPECT_NEAR(to_float(b + a), to_float(a) + to_float(b), tol);

    EXPECT_NEAR(to_float(a - b), to_float(a) - to_float(b), tol);
    EXPECT_NEAR(to_float(b - a), to_float(b) - to_float(a), tol);

    EXPECT_NEAR(to_float(a * b), to_float(a) * to_float(b), tol);
    EXPECT_NEAR(to_float(b * a), to_float(a) * to_float(b), tol);

    EXPECT_NEAR(to_float(a / b), to_float(a) / to_float(b), tol);
    EXPECT_NEAR(to_float(b / a), to_float(b) / to_float(a), tol);

    auto c = a;
    c += b;
    EXPECT_NEAR(to_float(c), to_float(a) + to_float(b), tol);
    c -= b;
    EXPECT_NEAR(to_float(c), to_float(a), tol);
    c *= b;
    EXPECT_NEAR(to_float(c), to_float(a) * to_float(b), tol);
    c /= b;
    EXPECT_NEAR(to_float(c), to_float(a), tol);
}

TEST(HalfTest, ConstructAndConvert) {
    float f = 1.5f;
    Half h(f);
    EXPECT_NEAR(static_cast<float>(h), f, 1e-3);

    Half h2(0.0f);
    EXPECT_EQ(static_cast<float>(h2), 0.0f);

    Half h3(-2.25f);
    EXPECT_NEAR(static_cast<float>(h3), -2.25f, 1e-3);
}

TEST(HalfTest, AllOps) {
    TestArithmeticOps<Half, Half>(Half(1.5f), Half(2.0f));

    TestArithmeticOps<Half, float>(Half(2.5f), 1.0f);
    TestArithmeticOps<float, Half>(1.0f, Half(2.5f));

    TestArithmeticOps<Half, int>(Half(3.0f), 2);
    TestArithmeticOps<int, Half>(2, Half(3.0f));
}

TEST(HalfTest, NumericLimits) {
    using Limits = std::numeric_limits<Half>;

    EXPECT_TRUE(Limits::is_specialized);
    EXPECT_TRUE(Limits::is_signed);
    EXPECT_FALSE(Limits::is_integer);

    EXPECT_NEAR(static_cast<float>(Limits::min()), 6.10352e-05f, 1e-8);
    EXPECT_LT(static_cast<float>(Limits::lowest()), 0.0f);
    EXPECT_GT(static_cast<float>(Limits::max()), 0.0f);

    EXPECT_TRUE(std::isinf(static_cast<float>(Limits::infinity())));
    EXPECT_TRUE(std::isnan(static_cast<float>(Limits::quiet_NaN())));
}

/// -------------------------- Device Test -------------------------------
template <typename T1, typename T2, typename Tout, typename Op>
void TestOpOnDevice(T1 a, T2 b, Op op) {
    Tout* d_out;
    cudaMalloc(&d_out, sizeof(Tout));

    auto kernel_lambda = [=] HOST_DEVICE (T1 a_, T2 b_, Tout* out) {
        *out = op(a_, b_);
    };

    LAUNCH_DEVICE_TEST(1, 1, kernel_lambda, a, b, d_out);

    Tout h_out;
    cudaMemcpy(&h_out, d_out, sizeof(Tout), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    Tout expected = op(a, b);
    EXPECT_NEAR(static_cast<float>(h_out), static_cast<float>(expected), 1e-3);
}

TEST(HalfCudaTest, AllOps) {
    // Half + Half -> Half
    TestOpOnDevice<Half, Half, Half>(Half(1.5f), Half(2.0f), AddOp());
    TestOpOnDevice<Half, Half, Half>(Half(1.5f), Half(2.0f), SubOp());
    TestOpOnDevice<Half, Half, Half>(Half(1.5f), Half(2.0f), MulOp());
    TestOpOnDevice<Half, Half, Half>(Half(3.0f), Half(1.5f), DivOp());

    // Half + float -> float
    TestOpOnDevice<Half, float, float>(Half(1.5f), 2.0f, AddOp());
    TestOpOnDevice<Half, float, float>(Half(1.5f), 2.0f, SubOp());
    TestOpOnDevice<Half, float, float>(Half(1.5f), 2.0f, MulOp());
    TestOpOnDevice<Half, float, float>(Half(3.0f), 1.5f, DivOp());

    // float + Half  -> float
    TestOpOnDevice<float, Half, float>(2.0f, Half(1.5f), AddOp());
    TestOpOnDevice<float, Half, float>(2.0f, Half(1.5f), SubOp());
    TestOpOnDevice<float, Half, float>(2.0f, Half(1.5f), MulOp());
    TestOpOnDevice<float, Half, float>(1.5f, Half(3.0f), DivOp());
}