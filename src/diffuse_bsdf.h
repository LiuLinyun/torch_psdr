#pragma once

#include "src/material.h"
#include "src/texture.h"

class DiffuseBsdfMaterial : public Material {
public:
  DiffuseBsdfMaterial() = delete;
  explicit DiffuseBsdfMaterial(
      std::shared_ptr<Texture3f> color = std::make_shared<Texture3f>(Vec3fD{
          0.8, 0.8, 0.8}),
      std::shared_ptr<Texture1f> roughness = std::make_shared<Texture1f>(Real{
          0.5}),
      std::shared_ptr<Texture3f> normal = nullptr)
      : m_color{color}, m_roughness{roughness}, m_tex_normal{normal} {}

  SampledBsdfC sample(const IntersectionC &its, const Vec2fC &sampler,
                      const MaskC &valid) const override;
  SampledBsdfD sample(const IntersectionD &its, const Vec2fD &sampler,
                      const MaskD &valid) const override;

  SpectrumC eval(const IntersectionC &its, const Vec3fC &wo,
                 const MaskC &valid) const override;
  SpectrumD eval(const IntersectionD &its, const Vec3fD &wo,
                 const MaskD &valid) const override;

  FloatC pdf(const IntersectionC &its, const Vec3fC &wo,
             const MaskC &valid) const override;
  FloatD pdf(const IntersectionD &its, const Vec3fD &wo,
             const MaskD &valid) const override;

private:
  template <bool ad>
  SampledBsdf<ad> sample_(const Intersection<ad> &its, const Vec2f<ad> &sampler,
                          const Mask<ad> &valid) const;

  template <bool ad>
  Spectrum<ad> eval_(const Intersection<ad> &its, const Vec3f<ad> &wo,
                     const Mask<ad> &valid) const;

  template <bool ad>
  Float<ad> pdf_(const Intersection<ad> &its, const Vec3f<ad> &wo,
                 const Mask<ad> &valid) const;

private:
  std::shared_ptr<Texture3f> m_color;
  std::shared_ptr<Texture1f> m_roughness;
  std::shared_ptr<Texture3f> m_tex_normal;
};