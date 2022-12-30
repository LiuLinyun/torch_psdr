#pragma once

#include "src/types.h"

template <typename TypeD>
requires enoki::is_diff_array_v<TypeD>
struct undiff {
  static TypeD val;
  // TODO: a more general way to trait type
  using type =
      std::conditional_t<std::is_same_v<TypeD, FloatD>, FloatC, Vec3fC>;
};

template <typename T> using undiff_t = typename undiff<T>::type;

template <typename TypeD> class Texture {
public:
  using TypeC = undiff_t<TypeD>;
  virtual ~Texture() = default;
  virtual TypeC at(const Vec2fC &uv, const MaskC &valid) const = 0;
  virtual TypeD at(const Vec2fD &uv, const MaskD &valid) const = 0;
};

template <typename TypeD> class ConstantTexture final : public Texture<TypeD> {
public:
  using TypeC = undiff_t<TypeD>;
  ConstantTexture() = delete;
  explicit ConstantTexture(const TypeD &val)
      : m_val{val} {};
  virtual TypeC at(const Vec2fC &uv, const MaskC &valid) const override {
    return enoki::select(valid, enoki::detach(m_val), 0);
  }
  virtual TypeD at(const Vec2fD &uv, const MaskD &valid) const override {
    return enoki::select(valid, m_val, 0);
  }

private:
  TypeD m_val;
};

template <typename TypeD> class UVTexture final : public Texture<TypeD> {
public:
  using TypeC = undiff_t<TypeD>;
  template <bool ad> using Type = typename std::conditional_t<ad, TypeD, TypeC>;
  UVTexture() = delete;
  UVTexture(uint32_t height, uint32_t width, const TypeD &data)
      : m_height{height}, m_width{width}, m_data{data} {
    auto cnt = enoki::slices(data);
    if (cnt != height * width) {
      // TODO throw exception
    }
  }

  uint32_t height() const { return m_height; }
  uint32_t width() const { return m_width; }

  virtual TypeC at(const Vec2fC &uv, const MaskC &valid) const override {
    return at_nearest<false>(enoki::detach(m_data), uv, valid);
  }

  virtual TypeD at(const Vec2fD &uv, const MaskD &valid) const override {
    return at_nearest<true>(m_data, uv, valid);
  }

  template <bool ad>
  Type<ad> at_nearest(const Type<ad> &data, const Vec2f<ad> &uv,
              const Mask<ad> &valid) const {
    auto u = uv.x() * m_width - Real{0.5};
    auto v = uv.y() * m_height - Real{0.5};
    auto u0 = enoki::clamp(Int<ad>{enoki::round(u)}, 0, m_width - 1);
    auto v0 = enoki::clamp(Int<ad>{enoki::round(v)}, 0, m_height - 1);
    auto idx = v0 * m_width + u0;
    return enoki::gather<Type<ad>>(data, idx, valid);
  }

  template <bool ad>
  Type<ad> at_bilinear(const Type<ad> &data, const Vec2f<ad> &uv,
              const Mask<ad> &valid) const {
    //  (u0,v1)---(u1,v1)
    //     |         |
    //  (u0,v0)---(u1,v0)
    auto u = uv.x() * m_width - Real{0.5};
    auto v = uv.y() * m_height - Real{0.5};
    auto u0 = enoki::clamp(enoki::floor2int<Int<ad>>(u), 0, m_width - 2);
    auto v0 = enoki::clamp(enoki::floor2int<Int<ad>>(v), 0, m_height - 2);
    auto get_idx_val = [&](const auto& u, const auto& v) -> auto {
      auto idx = v * m_width + u;
      return enoki::gather<Type<ad>>(data, idx, valid);
    };
    auto u0v0 = get_idx_val(u0,v0);
    auto u0v1 = get_idx_val(u0,v0+1);
    auto u1v0 = get_idx_val(u0+1,v0);
    auto u1v1 = get_idx_val(u0+1,v0+1);
    auto tmp_v0 = enoki::lerp(u0v0, u1v0, u - u0);
    auto tmp_v1 = enoki::lerp(u0v1, u1v1, u - u0);
    auto rst = enoki::lerp(tmp_v0, tmp_v1, v - v0);
    return rst;
  }

private:
  uint32_t m_height;
  uint32_t m_width;
  TypeD m_data;
};

using Texture1f = Texture<FloatD>;
using Texture3f = Texture<Vec3fD>;
using ConstantTexture1f = ConstantTexture<FloatD>;
using ConstantTexture3f = ConstantTexture<Vec3fD>;
using UVTexture1f = UVTexture<FloatD>;
using UVTexture3f = UVTexture<Vec3fD>;
