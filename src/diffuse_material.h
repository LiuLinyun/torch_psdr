#pragma once

#include "src/material.h"
#include "src/texture.h"

class DiffuseMaterial : public Material {
public:
  DiffuseMaterial() = delete;
  explicit DiffuseMaterial(const Vec3fD &tex)
      : m_tex{std::make_shared<ConstantTexture3f>(tex)} {}
  explicit DiffuseMaterial(std::shared_ptr<Texture3f> tex) : m_tex{tex} {}

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
  SampledBsdf<ad> sample_(const Intersection<ad> &its,
                           const Vec2f<ad> &sampler,
                           const Mask<ad> &valid) const;

  template <bool ad>
  Spectrum<ad> eval_(const Intersection<ad> &its, const Vec3f<ad> &wo,
                     const Mask<ad> &valid) const;

  template <bool ad>
  Float<ad> pdf_(const Intersection<ad> &its, const Vec3f<ad> &wo,
                 const Mask<ad> &valid) const;

protected:
  std::shared_ptr<Texture3f> m_tex;
};

class DiffuseLightMaterial : public DiffuseMaterial {
public:
  DiffuseLightMaterial() = delete;
  explicit DiffuseLightMaterial(const Vec3fD &tex) : DiffuseMaterial{tex} {}
  explicit DiffuseLightMaterial(std::shared_ptr<Texture3f> tex)
      : DiffuseMaterial{tex} {}

  SpectrumC emit(const IntersectionC &its, const MaskC &valid) const override {
    return m_tex->at(its.uv, valid);
  }
  SpectrumD emit(const IntersectionD &its, const MaskD &valid) const override {
    return m_tex->at(its.uv, valid);
  }
};
