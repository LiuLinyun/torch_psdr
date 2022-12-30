#pragma once

#include <array>
#include <numbers>
#include <vector>
#include <memory>

#include "enoki/array.h"
#include "enoki/array_macro.h"
#include "enoki/array_struct.h"
#include "enoki/autodiff.h"
#include "enoki/cuda.h"
#include "enoki/matrix.h"
#include "enoki/transform.h"
#include "enoki/stl.h"

using Real = float;
using Integer = int32_t;
constexpr auto SPECTRUM_CHANNELS = 3;

template <typename T, bool ad>
using Type =
    typename std::conditional<ad, enoki::DiffArray<enoki::CUDAArray<T>>,
                              enoki::CUDAArray<T>>::type;

template <typename T> using TypeC = Type<T, false>;

template <typename T> using TypeD = Type<T, true>;

template <typename T, size_t n, bool ad>
using Vec = enoki::Array<Type<T, ad>, n>;

template <typename T, size_t n> using VecC = Vec<T, n, false>;

template <typename T, size_t n> using VecD = Vec<T, n, true>;

template <bool ad> using Float = Type<Real, ad>;

using FloatC = Float<false>;
using FloatD = Float<true>;

template <bool ad> using Int = Type<Integer, ad>;

using IntC = Int<false>;
using IntD = Int<true>;

template <typename T> using Vec2 = enoki::Array<T, 2>;

template <typename T> using Vec3 = enoki::Array<T, 3>;

template <typename T> using Vec4 = enoki::Array<T, 4>;

template <size_t n, bool ad> using Vecf = Vec<Real, n, ad>;

template <size_t n, bool ad> using Veci = Vec<Integer, n, ad>;

template <bool ad> using Vec1f = Vecf<1, ad>;

template <bool ad> using Vec1i = Veci<1, ad>;

template <bool ad> using Vec2f = Vecf<2, ad>;

template <bool ad> using Vec2i = Veci<2, ad>;

template <bool ad> using Vec3f = Vecf<3, ad>;

template <bool ad> using Vec3i = Veci<3, ad>;

template <bool ad> using Vec4f = Vecf<4, ad>;

template <bool ad> using Vec4i = Veci<4, ad>;

template <bool ad> using Mat3f = enoki::Matrix<Float<ad>, 3>;

template <bool ad> using Mat4f = enoki::Matrix<Float<ad>, 4>;

using Vec1fC = Vec1f<false>;
using Vec1fD = Vec1f<true>;
using Vec2fC = Vec2f<false>;
using Vec2fD = Vec2f<true>;
using Vec3fC = Vec3f<false>;
using Vec3fD = Vec3f<true>;
using Vec4fC = Vec4f<false>;
using Vec4fD = Vec4f<true>;

using Vec1iC = Vec1i<false>;
using Vec1iD = Vec1i<true>;
using Vec2iC = Vec2i<false>;
using Vec2iD = Vec2i<true>;
using Vec3iC = Vec3i<false>;
using Vec3iD = Vec3i<true>;
using Vec4iC = Vec4i<false>;
using Vec4iD = Vec4i<true>;

using Mat3fC = Mat3f<false>;
using Mat3fD = Mat3f<true>;
using Mat4fC = Mat4f<false>;
using Mat4fD = Mat4f<true>;

template <bool ad> using Mask = enoki::mask_t<Float<ad>>;

using MaskC = Mask<false>;
using MaskD = Mask<true>;

// for PCG32 sampler
using UIntC = Type<uint32_t, false>;
using UInt64C = Type<uint64_t, false>;

template <bool ad> using Spectrum = Vecf<SPECTRUM_CHANNELS, ad>;

using SpectrumC = Spectrum<false>;
using SpectrumD = Spectrum<true>;

constexpr auto REAL_EPS = Real{1e-5};
constexpr auto REAL_MIN = std::numeric_limits<Real>::min();
constexpr auto REAL_MAX = std::numeric_limits<Real>::max();
constexpr auto REAL_INF = std::numeric_limits<Real>::infinity();
constexpr auto REAL_PI = Real{std::numbers::pi};
constexpr auto REAL_2PI = Real{2 * std::numbers::pi};
constexpr auto REAL_4PI = Real{4 * std::numbers::pi};
constexpr auto REAL_INV_PI = Real{1 / std::numbers::pi};
constexpr auto REAL_INV_2PI = Real{1 / (2 * std::numbers::pi)};
constexpr auto REAL_INV_4PI = Real{1 / (4 * std::numbers::pi)};
