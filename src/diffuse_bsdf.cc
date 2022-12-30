#include "src/diffuse_bsdf.h"

template <bool ad>
SampledBsdf<ad> DiffuseBsdfMaterial::sample_(const Intersection<ad> &its,
                                             const Vec2f<ad> &sampler,
                                             const Mask<ad> &valid) const {
  auto &uv = its.uv;
  auto &V = its.wi;
  auto &Ng = its.normal;
  auto new_valid = valid && its.valid && enoki::dot(Ng, V) > 0;
  auto &T = its.tangent;
  auto B = enoki::cross(Ng, T);
  auto N = Vec3f<ad>{};
  if (m_tex_normal == nullptr) {
    N = Ng;
  } else {
    auto tex_normal = m_tex_normal->at(uv, new_valid);
    auto TBN = Mat3f<ad>{T, B, Ng};
    N = enoki::normalize(TBN * (tex_normal * 2 - 1));
  }

  auto &randu = sampler.x();
  auto &randv = sampler.y();
  auto cos_theta = enoki::safe_sqrt(randu);
  auto sin_theta = enoki::safe_sqrt(1 - randu);
  auto phi = randv * REAL_2PI;
  auto [sin_phi, cos_phi] = enoki::sincos(phi);
  auto u = cos_phi * sin_theta;
  auto v = sin_phi * sin_theta;
  auto L = enoki::normalize(u * T + v * B + cos_theta * N);
  auto pdf = cos_theta * REAL_INV_PI;

  auto color = m_color->at(uv, new_valid);
  auto roughness = m_roughness->at(uv, new_valid);
  auto &sigma = roughness;
  auto div = 1 / (REAL_PI + ((3 * REAL_PI - 4) / 6) * sigma);
  auto a = div;
  auto b = sigma * div;
  auto NdotL = enoki::max(enoki::dot(N, L), 0);
  auto NdotV = enoki::max(enoki::dot(N, V), 0);
  auto VdotL = enoki::dot(V, L);
  auto t = VdotL - NdotL * NdotV;
  t[t > 0] = enoki::max(NdotL, NdotV) + REAL_EPS;
  auto bsdf = color * (a + b * t);

  auto rst = SampledBsdf<ad>{};
  rst.wo = std::move(L);
  rst.pdf = std::move(pdf);
  rst.bsdf = std::move(bsdf);
  rst.valid = std::move(new_valid);
  return rst;
}

template <bool ad>
Spectrum<ad> DiffuseBsdfMaterial::eval_(const Intersection<ad> &its,
                                        const Vec3f<ad> &wo,
                                        const Mask<ad> &valid) const {
  auto &uv = its.uv;

  auto &V = its.wi;
  auto &L = wo;
  auto &Ng = its.normal;
  auto &T = its.tangent;
  auto B = enoki::cross(Ng, T);
  auto N = Vec3f<ad>{};
  if (m_tex_normal == nullptr) {
    N = Ng;
  } else {
    auto &uv = its.uv;
    auto tex_normal = m_tex_normal->at(uv, valid);
    auto TBN = Mat3f<ad>{T, B, Ng};
    N = enoki::normalize(TBN * (tex_normal * 2 - 1));
  }

  auto color = m_color->at(uv, valid);
  auto roughness = m_roughness->at(uv, valid);
  auto &sigma = roughness;
  auto div = 1 / (REAL_PI + ((3 * REAL_PI - 4) / 6) * sigma);
  auto a = div;
  auto b = sigma * div;
  auto NdotL = enoki::max(enoki::dot(N, L), 0);
  auto NdotV = enoki::max(enoki::dot(N, V), 0);
  auto VdotL = enoki::dot(V, L);
  auto t = VdotL - NdotL * NdotV;
  t[t > 0] = enoki::max(NdotL, NdotV) + REAL_EPS;
  auto bsdf = color * (a + b * t);
  return bsdf;
}

template <bool ad>
Float<ad> DiffuseBsdfMaterial::pdf_(const Intersection<ad> &its,
                                    const Vec3f<ad> &wo,
                                    const Mask<ad> &valid) const {
  auto cos_theta = enoki::dot(its.normal, wo);
  return enoki::select(its.valid && valid, cos_theta * REAL_INV_PI, 0);
}

SampledBsdfC DiffuseBsdfMaterial::sample(const IntersectionC &its,
                                         const Vec2fC &sampler,
                                         const MaskC &valid) const {
  return sample_<false>(its, sampler, valid);
}
SampledBsdfD DiffuseBsdfMaterial::sample(const IntersectionD &its,
                                         const Vec2fD &sampler,
                                         const MaskD &valid) const {
  return sample_<true>(its, sampler, valid);
}

SpectrumC DiffuseBsdfMaterial::eval(const IntersectionC &its, const Vec3fC &wo,
                                    const MaskC &valid) const {
  return eval_<false>(its, wo, valid);
}
SpectrumD DiffuseBsdfMaterial::eval(const IntersectionD &its, const Vec3fD &wo,
                                    const MaskD &valid) const {
  return eval_<true>(its, wo, valid);
}

FloatC DiffuseBsdfMaterial::pdf(const IntersectionC &its, const Vec3fC &wo,
                                const MaskC &valid) const {
  return pdf_<false>(its, wo, valid);
}
FloatD DiffuseBsdfMaterial::pdf(const IntersectionD &its, const Vec3fD &wo,
                                const MaskD &valid) const {
  return pdf_<true>(its, wo, valid);
}