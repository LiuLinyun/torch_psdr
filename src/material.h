#pragma once

#include "src/intersection.h"
#include "src/ray.h"
#include "src/sampler.h"
#include "src/texture.h"
#include "src/types.h"

template <typename Value> struct SampledBsdf_ {
  Vec3<Value> wo;
  Value pdf;
  Vec3<Value> bsdf;
  enoki::mask_t<Value> valid;

  ENOKI_STRUCT(SampledBsdf_, wo, pdf, bsdf, valid)
};
ENOKI_STRUCT_SUPPORT(SampledBsdf_, wo, pdf, bsdf, valid)

template <bool ad> using SampledBsdf = SampledBsdf_<Float<ad>>;

using SampledBsdfC = SampledBsdf<false>;
using SampledBsdfD = SampledBsdf<true>;

class Material {
public:
  virtual ~Material() = default;

  virtual SampledBsdfC sample(const IntersectionC &its, const Vec2fC &sampler,
                               const MaskC &valid = true) const = 0;
  virtual SampledBsdfD sample(const IntersectionD &its, const Vec2fD &sampler,
                               const MaskD &valid = true) const = 0;

  virtual SpectrumC eval(const IntersectionC &its, const Vec3fC &wo,
                         const MaskC &valid = true) const = 0;
  virtual SpectrumD eval(const IntersectionD &its, const Vec3fD &wo,
                         const MaskD &valid = true) const = 0;

  virtual FloatC pdf(const IntersectionC &its, const Vec3fC &wo,
                     const MaskC &valid = true) const = 0;
  virtual FloatD pdf(const IntersectionD &its, const Vec3fD &wo,
                     const MaskD &valid = true) const = 0;

  virtual SpectrumC emit(const IntersectionC &its,
                         const MaskC &valid = true) const {
    return SpectrumC{0};
  }
  virtual SpectrumD emit(const IntersectionD &its,
                         const MaskD &valid = true) const {
    return SpectrumD{0};
  }
};

ENOKI_CALL_SUPPORT_BEGIN(Material)
ENOKI_CALL_SUPPORT_METHOD(sample)
ENOKI_CALL_SUPPORT_METHOD(eval)
ENOKI_CALL_SUPPORT_METHOD(pdf)
ENOKI_CALL_SUPPORT_METHOD(emit)
ENOKI_CALL_SUPPORT_END(Material)
