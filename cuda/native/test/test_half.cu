#include <cuda_runtime.h>

#include "mini_gtest.h"
#include "device_launch.cuh"
#include "half.h"

using witin::Half;

TEST(HalfTest, ConstructAndConvert) {
    float f = 1.5f;
    Half h(f);
    EXPECT_NEAR(static_cast<float>(h), f, 1e-3);

    Half h2(0.0f);
    EXPECT_EQ(static_cast<float>(h2), 0.0f);

    Half h3(-2.25f);
    EXPECT_NEAR(static_cast<float>(h3), -2.25f, 1e-3);
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

struct CompareOp {
    template<typename A, typename B> HOST_DEVICE bool lt(A a, B b) const { return a <  b; }
    template<typename A, typename B> HOST_DEVICE bool le(A a, B b) const { return a <= b; }
    template<typename A, typename B> HOST_DEVICE bool gt(A a, B b) const { return a >  b; }
    template<typename A, typename B> HOST_DEVICE bool ge(A a, B b) const { return a >= b; }
    template<typename A, typename B> HOST_DEVICE bool eq(A a, B b) const { return a == b; }
    template<typename A, typename B> HOST_DEVICE bool ne(A a, B b) const { return a != b; }
};

struct AddOp { template<typename A, typename B> HOST_DEVICE auto operator()(A a, B b) const { return a + b; } };
struct SubOp { template<typename A, typename B> HOST_DEVICE auto operator()(A a, B b) const { return a - b; } };
struct MulOp { template<typename A, typename B> HOST_DEVICE auto operator()(A a, B b) const { return a * b; } };
struct DivOp { template<typename A, typename B> HOST_DEVICE auto operator()(A a, B b) const { return a / b; } };

/// -------------------------- Host Test -------------------------------
template <typename T1, typename T2, typename Tout, typename func_t>
void TestBinaryOpsHost(T1 a, T2 b, Tout expect, func_t op) {
    Tout out = op(a, b);
    EXPECT_EQ(out, expect);
}

TEST(HalfTest, CompareOps) {
    CompareOp cmp;

    // Half vs Half
    TestBinaryOpsHost<Half, Half>(Half(1.5f), Half(2.0f), true,  [&](auto x, auto y){ return cmp.lt(x, y); });
    TestBinaryOpsHost<Half, Half>(Half(1.5f), Half(2.0f), false, [&](auto x, auto y){ return cmp.ge(x, y); });

    // Half vs float
    TestBinaryOpsHost<Half, float>(Half(2.5f), 1.0f, true,  [&](auto x, auto y){ return cmp.gt(x, y); });
    TestBinaryOpsHost<Half, float>(Half(2.5f), 1.0f, false, [&](auto x, auto y){ return cmp.le(x, y); });

    // float vs Half
    TestBinaryOpsHost<float, Half>(1.0f, Half(2.5f), true,  [&](auto x, auto y){ return cmp.lt(x, y); });
    TestBinaryOpsHost<float, Half>(1.0f, Half(2.5f), false, [&](auto x, auto y){ return cmp.ge(x, y); });

    // Half vs int
    TestBinaryOpsHost<Half, int>(Half(3.0f), 2, false, [&](auto x, auto y){ return cmp.lt(x, y); });
    TestBinaryOpsHost<Half, int>(Half(3.0f), 2, true,  [&](auto x, auto y){ return cmp.gt(x, y); });

    // int vs Half
    TestBinaryOpsHost<int, Half>(2, Half(3.0f), true,  [&](auto x, auto y){ return cmp.lt(x, y); });
    TestBinaryOpsHost<int, Half>(2, Half(3.0f), false, [&](auto x, auto y){ return cmp.ge(x, y); });
}

TEST(HalfTest, AllOps) {
    // Half + Half -> Half
    TestBinaryOpsHost<Half, Half, Half>(Half(1.5f), Half(2.0f), Half(3.5f),  AddOp());
    TestBinaryOpsHost<Half, Half, Half>(Half(1.5f), Half(2.0f), Half(-0.5f), SubOp());
    TestBinaryOpsHost<Half, Half, Half>(Half(1.5f), Half(2.0f), Half(3.0f),  MulOp());
    TestBinaryOpsHost<Half, Half, Half>(Half(3.0f), Half(1.5f), Half(2.0f),  DivOp());

    // Half + float -> float
    TestBinaryOpsHost<Half, float, float>(Half(1.5f), 2.0f, 3.5f,  AddOp());
    TestBinaryOpsHost<Half, float, float>(Half(1.5f), 2.0f, -0.5f, SubOp());
    TestBinaryOpsHost<Half, float, float>(Half(1.5f), 2.0f, 3.0f,  MulOp());
    TestBinaryOpsHost<Half, float, float>(Half(3.0f), 1.5f, 2.0f,  DivOp());

    // float + Half  -> float
    TestBinaryOpsHost<float, Half, float>(2.0f, Half(1.5f), 3.5f, AddOp());
    TestBinaryOpsHost<float, Half, float>(2.0f, Half(1.5f), 0.5f, SubOp());
    TestBinaryOpsHost<float, Half, float>(2.0f, Half(1.5f), 3.0f, MulOp());
    TestBinaryOpsHost<float, Half, float>(1.5f, Half(3.0f), 0.5f, DivOp());
}

/// -------------------------- Device Test -------------------------------
template <typename T1, typename T2, typename Tout, typename Op>
void TestBinaryOpDevice(T1 a, T2 b, Op op) {
    Tout* d_out;
    cudaMalloc(&d_out, sizeof(Tout));

    auto kernel_lambda = [=] HOST_DEVICE (T1 a_, T2 b_, Tout* out) {
        *out = op(a_, b_);
    };

    LAUNCH_DEVICE_TEST(1, 1, kernel_lambda, a, b, d_out);

    Tout h_out;
    cudaMemcpy(&h_out, d_out, sizeof(Tout), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    Tout expect = op(a, b);
    EXPECT_EQ(h_out, expect);
}

TEST(HalfCudaTest, AllOps) {
    // Half + Half -> Half
    TestBinaryOpDevice<Half, Half, Half>(Half(1.5f), Half(2.0f), AddOp());
    TestBinaryOpDevice<Half, Half, Half>(Half(1.5f), Half(2.0f), SubOp());
    TestBinaryOpDevice<Half, Half, Half>(Half(1.5f), Half(2.0f), MulOp());
    TestBinaryOpDevice<Half, Half, Half>(Half(3.0f), Half(1.5f), DivOp());

    // Half + float -> float
    TestBinaryOpDevice<Half, float, float>(Half(1.5f), 2.0f, AddOp());
    TestBinaryOpDevice<Half, float, float>(Half(1.5f), 2.0f, SubOp());
    TestBinaryOpDevice<Half, float, float>(Half(1.5f), 2.0f, MulOp());
    TestBinaryOpDevice<Half, float, float>(Half(3.0f), 1.5f, DivOp());

    // float + Half  -> float
    TestBinaryOpDevice<float, Half, float>(2.0f, Half(1.5f), AddOp());
    TestBinaryOpDevice<float, Half, float>(2.0f, Half(1.5f), SubOp());
    TestBinaryOpDevice<float, Half, float>(2.0f, Half(1.5f), MulOp());
    TestBinaryOpDevice<float, Half, float>(1.5f, Half(3.0f), DivOp());
}