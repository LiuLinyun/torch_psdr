#include "src/diffuse_material.h"

template <bool ad>
SampledBsdf<ad> DiffuseMaterial::sample_(const Intersection<ad> &its,
                                         const Vec2f<ad> &sampler,
                                         const Mask<ad> &valid) const {
  auto &N = its.normal;
  auto &T = its.tangent;
  auto B = enoki::cross(N, T);

  auto &randu = sampler.x();
  auto &randv = sampler.y();
  auto cos_theta = enoki::safe_sqrt(randu);
  auto sin_theta = enoki::safe_sqrt(1 - randu);
  auto phi = randv * REAL_2PI;
  auto [sin_phi, cos_phi] = enoki::sincos(phi);
  auto u = cos_phi * sin_theta;
  auto v = sin_phi * sin_theta;
  auto L = enoki::normalize(u * T + v * B + cos_theta * N);

  auto new_valid = valid && its.valid;
  auto bsdf = m_tex->at(its.uv, new_valid) * REAL_INV_PI;

  auto sampled = SampledBsdf<ad>{};
  sampled.wo = L;
  sampled.valid = new_valid;
  sampled.pdf = cos_theta * REAL_INV_PI;
  sampled.bsdf = bsdf;
  return sampled;
}

template <bool ad>
Spectrum<ad> DiffuseMaterial::eval_(const Intersection<ad> &its,
                                    const Vec3f<ad> &wo,
                                    const Mask<ad> &valid) const {
  return m_tex->at(its.uv, valid && its.valid) * REAL_INV_PI;
}

template <bool ad>
Float<ad> DiffuseMaterial::pdf_(const Intersection<ad> &its,
                                const Vec3f<ad> &wo,
                                const Mask<ad> &valid) const {
  auto cos_theta = enoki::dot(its.normal, wo);
  return enoki::select(valid && its.valid, cos_theta * REAL_INV_PI, 0);
}

SampledBsdfC DiffuseMaterial::sample(const IntersectionC &its,
                                     const Vec2fC &sampler,
                                     const MaskC &valid) const {
  return sample_<false>(its, sampler, valid);
}
SampledBsdfD DiffuseMaterial::sample(const IntersectionD &its,
                                     const Vec2fD &sampler,
                                     const MaskD &valid) const {
  return sample_<true>(its, sampler, valid);
}

SpectrumC DiffuseMaterial::eval(const IntersectionC &its, const Vec3fC &wo,
                                const MaskC &valid) const {
  return eval_<false>(its, wo, valid);
}
SpectrumD DiffuseMaterial::eval(const IntersectionD &its, const Vec3fD &wo,
                                const MaskD &valid) const {
  return eval_<true>(its, wo, valid);
}

FloatC DiffuseMaterial::pdf(const IntersectionC &its, const Vec3fC &wo,
                            const MaskC &valid) const {
  return pdf_<false>(its, wo, valid);
}
FloatD DiffuseMaterial::pdf(const IntersectionD &its, const Vec3fD &wo,
                            const MaskD &valid) const {
  return pdf_<true>(its, wo, valid);
}
