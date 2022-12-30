#pragma once

#include "src/diffuse_material.h"
#include "src/mesh.h"

enum class LightType {
  area,
  point,
  ambient,
  envmap,
  spherical_harmonics,  // 球谐
};

template <typename Value>
struct SampledLightPoint_ {
  Vec3<Value> point;
  Value pdf;
  ENOKI_STRUCT(SampledLightPoint_, point, pdf)
};
ENOKI_STRUCT_SUPPORT(SampledLightPoint_, point, pdf)

template <bool ad>
using SampledLightPoint = SampledLightPoint_<Float<ad>>;
using SampledLightPointC = SampledLightPoint<false>;
using SampledLightPointD = SampledLightPoint<true>;

template <bool ad>
ENOKI_INLINE Float<ad> rgb2luminance(const Vec3f<ad>& rgb) {
  return rgb.x() * Real{.2126} + rgb.y() * Real{.7152} + rgb.z() * Real{.0722};
}

template <bool ad>
ENOKI_INLINE Float<ad> mis_weight(const Float<ad>& w1, const Float<ad>& w2) {
  auto sqr_w1 = w1 * w1;
  auto sqr_w2 = w2 * w2;
  return sqr_w1 / (sqr_w1 + sqr_w2);
}

class Light {
 public:
  Light(LightType type) : m_light_type{type} {};
  virtual ~Light() = default;
  LightType type() const { return m_light_type; }

  virtual void set_aabb(const Vec3fC& pmin, const Vec3fC& pmax) {
    ;  // default do nothing, only env map light need set aabb
  }

  virtual Real power() const = 0;

  virtual SpectrumC eval(const IntersectionC& its,
                         const MaskC& valid = true) const = 0;
  virtual SpectrumD eval(const IntersectionD& its,
                         const MaskD& valid = true) const = 0;

  virtual SampledLightPointC sample_light_point(const Vec2fC& sampler,
                                                const MaskC& valid) const = 0;
  virtual SampledLightPointD sample_light_point(const Vec2fD& sampler,
                                                const MaskD& valid) const = 0;

  virtual FloatC pdf_sampled_light_point(const Vec3fC& point,
                                         const MaskC& valid) const = 0;
  virtual FloatD pdf_sampled_light_point(const Vec3fD& point,
                                         const MaskD& valid) const = 0;

  virtual std::shared_ptr<TriangleMesh> mesh() const = 0;

 private:
  LightType m_light_type;
};

ENOKI_CALL_SUPPORT_BEGIN(Light)
ENOKI_CALL_SUPPORT_METHOD(power)
ENOKI_CALL_SUPPORT_METHOD(eval)
ENOKI_CALL_SUPPORT_METHOD(sample_light_point)
ENOKI_CALL_SUPPORT_METHOD(pdf_sampled_light_point)
ENOKI_CALL_SUPPORT_END(Light)

class AreaLight : public Light {
 public:
  AreaLight() = delete;
  AreaLight(const Vec3fD& vertices, const Vec3iC& face_indices,
            const Vec3fD& emit);

  Real power() const override { return m_power[0]; }

  SpectrumC eval(const IntersectionC& its,
                 const MaskC& valid = true) const override;
  SpectrumD eval(const IntersectionD& its,
                 const MaskD& valid = true) const override;

  SampledLightPointC sample_light_point(const Vec2fC& sampler,
                                        const MaskC& valid) const override;
  SampledLightPointD sample_light_point(const Vec2fD& sampler,
                                        const MaskD& valid) const override;

  FloatC pdf_sampled_light_point(const Vec3fC& point,
                                 const MaskC& valid) const override;
  FloatD pdf_sampled_light_point(const Vec3fD& point,
                                 const MaskD& valid) const override;

  std::shared_ptr<TriangleMesh> mesh() const override { return m_mesh; }

 private:
  template <bool ad>
  Spectrum<ad> eval_(const Intersection<ad>& its, const Mask<ad>& valid) const;

  template <bool ad>
  SampledLightPoint<ad> sample_light_point_(const Vec2f<ad>& sampler,
                                            const Mask<ad>& valid) const;

  template <bool ad>
  Float<ad> pdf_sampled_light_point_(const Vec3f<ad>& point,
                                     const Mask<ad>& valid) const;

  Vec3fD m_emit;  // Radiance
  std::shared_ptr<TriangleMesh> m_mesh;
  FloatC m_power;
  std::shared_ptr<AliasSampler> m_triangle_sampler;
};

class EnvLight : public Light {
 public:
  EnvLight() = delete;
  explicit EnvLight(std::shared_ptr<UVTexture3f> cube_map_tex);
  void set_aabb(const Vec3fC& pmin, const Vec3fC& pmax) override;
  Real power() const override { return m_power[0]; }

  SpectrumC eval(const IntersectionC& its,
                 const MaskC& valid = true) const override;
  SpectrumD eval(const IntersectionD& its,
                 const MaskD& valid = true) const override;

  SampledLightPointC sample_light_point(const Vec2fC& sampler,
                                        const MaskC& valid) const override;
  SampledLightPointD sample_light_point(const Vec2fD& sampler,
                                        const MaskD& valid) const override;

  FloatC pdf_sampled_light_point(const Vec3fC& point,
                                 const MaskC& valid) const override;
  FloatD pdf_sampled_light_point(const Vec3fD& point,
                                 const MaskD& valid) const override;

  std::shared_ptr<TriangleMesh> mesh() const override { return m_mesh; }

 private:
  template <bool ad>
  Spectrum<ad> eval_(const Intersection<ad>& its, const Mask<ad>& valid) const;
  template <bool ad>
  SampledLightPoint<ad> sample_light_point_(const Vec2f<ad>& sampler,
                                            const Mask<ad>& valid) const;
  template <bool ad>
  Float<ad> pdf_sampled_light_point_(const Vec3f<ad>& point,
                                     const Mask<ad>& valid) const;

  std::shared_ptr<TriangleMesh> m_mesh;
  std::shared_ptr<UVTexture3f> m_cube_map_tex;
  std::shared_ptr<AliasSampler> m_triangle_sampler;
  FloatC m_power;
};

class SH9Light : public Light {
 public:
  template <bool ad>
  using Coeff = enoki::Array<Float<ad>, 9>;

  template <bool ad>
  using Coeff3f = enoki::Array<Vec3f<ad>, 9>;

  SH9Light() = delete;
  explicit SH9Light(const Vec3fD& sh9_coeff);
  void set_aabb(const Vec3fC& pmin, const Vec3fC& pmax) override;

  Real power() const override { return m_power[0]; }

  SpectrumC eval(const IntersectionC& its,
                 const MaskC& valid = true) const override;
  SpectrumD eval(const IntersectionD& its,
                 const MaskD& valid = true) const override;

  SampledLightPointC sample_light_point(const Vec2fC& sampler,
                                        const MaskC& valid) const override;
  SampledLightPointD sample_light_point(const Vec2fD& sampler,
                                        const MaskD& valid) const override;

  FloatC pdf_sampled_light_point(const Vec3fC& point,
                                 const MaskC& valid) const override;
  FloatD pdf_sampled_light_point(const Vec3fD& point,
                                 const MaskD& valid) const override;

  std::shared_ptr<TriangleMesh> mesh() const override { return m_mesh; }

 private:
  template <bool ad>
  Spectrum<ad> eval_impl_(const Vec3f<ad>& dir, const Mask<ad>& valid) const;
  template <bool ad>
  Spectrum<ad> eval_(const Intersection<ad>& its, const Mask<ad>& valid) const;
  template <bool ad>
  SampledLightPoint<ad> sample_light_point_(const Vec2f<ad>& sampler,
                                            const Mask<ad>& valid) const;
  template <bool ad>
  Float<ad> pdf_sampled_light_point_(const Vec3f<ad>& point,
                                     const Mask<ad>& valid) const;

 private:
  Coeff3f<true> m_sh9_coeff;
  std::shared_ptr<TriangleMesh> m_mesh;
  std::shared_ptr<AliasSampler> m_triangle_sampler;
  FloatC m_power;
};