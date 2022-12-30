#include "src/emitter.h"

///////////////////// area light ///////////////////
AreaLight::AreaLight(const Vec3fD &vertices, const Vec3iC &face_indices,
                     const Vec3fD &emit)
    : Light{LightType::area}, m_emit{emit},
      m_mesh{std::make_shared<TriangleMesh>(
          vertices, Vec2fC{}, face_indices, Vec3iC{},
          std::make_shared<DiffuseLightMaterial>(emit))} {
  m_power = REAL_2PI * rgb2luminance<false>(enoki::detach(emit)) *
            enoki::detach(m_mesh->area());
  m_triangle_sampler =
      std::make_shared<AliasSampler>(enoki::detach(m_mesh->triangles().area));
}

template <bool ad>
Spectrum<ad> AreaLight::eval_(const Intersection<ad> &its,
                              const Mask<ad> &valid) const {
  auto new_valid = its.valid && valid;
  auto emit = Spectrum<ad>{};
  if constexpr (ad) {
    emit = m_emit;
  } else {
    emit = enoki::detach(m_emit);
  }
  return enoki::select(new_valid, emit, 0);
}

template <bool ad>
SampledLightPoint<ad>
AreaLight::sample_light_point_(const Vec2f<ad> &sampler_,
                               const Mask<ad> &valid) const {
  auto sampler = sampler_;
  auto [idx, _] = m_triangle_sampler->sample_reuse<ad>(sampler.x(), valid);

  auto &triangles = m_mesh->triangles();
  auto sampled_triangles = Triangle<ad>{};
  if constexpr (ad) {
    sampled_triangles = enoki::gather<TriangleD>(triangles, idx, valid);
  } else {
    sampled_triangles =
        enoki::gather<TriangleC>(enoki::detach(triangles), idx, valid);
  }

  // p = sqrt(v)*(1-u) * A + u*sqrt(v) * B + (1-sqrt(v)) * C
  //   = A + u*sqrt(v)*(B-A) + (1-sqrt(v))*(C-A)
  auto &randu = sampler.x();
  auto &randv = sampler.y();
  auto sqrt_v = enoki::safe_sqrt(randv);
  auto u = randu * sqrt_v;
  auto v = 1 - sqrt_v;

  auto rst = SampledLightPoint<ad>{};
  rst.point = sampled_triangles.p0 +
              (sampled_triangles.p1 - sampled_triangles.p0) * u +
              (sampled_triangles.p2 - sampled_triangles.p0) * v;
  rst.pdf = enoki::rcp(enoki::detach(m_mesh->area()));
  return rst;
}

template <bool ad>
Float<ad> AreaLight::pdf_sampled_light_point_(const Vec3f<ad> &point,
                                              const Mask<ad> &valid) const {
  return enoki::select(valid, enoki::rcp(enoki::detach(m_mesh->area())), 0);
}

SpectrumC AreaLight::eval(const IntersectionC &its, const MaskC &valid) const {
  return eval_<false>(its, valid);
}
SpectrumD AreaLight::eval(const IntersectionD &its, const MaskD &valid) const {
  return eval_<true>(its, valid);
}

SampledLightPointC AreaLight::sample_light_point(const Vec2fC &sampler,
                                                 const MaskC &valid) const {
  return sample_light_point_<false>(sampler, valid);
}

SampledLightPointD AreaLight::sample_light_point(const Vec2fD &sampler,
                                                 const MaskD &valid) const {
  return sample_light_point_<true>(sampler, valid);
}

FloatC AreaLight::pdf_sampled_light_point(const Vec3fC &point,
                                          const MaskC &valid) const {
  return pdf_sampled_light_point_<false>(point, valid);
}

FloatD AreaLight::pdf_sampled_light_point(const Vec3fD &point,
                                          const MaskD &valid) const {
  return pdf_sampled_light_point_<true>(point, valid);
}

//////////////////// env light /////////////////////
EnvLight::EnvLight(std::shared_ptr<UVTexture3f> cube_map_tex)
    : Light{LightType::envmap}, m_cube_map_tex{cube_map_tex} {}

void EnvLight::set_aabb(const Vec3fC &pmin, const Vec3fC &pmax) {
  // construct mesh
  auto [x0, y0, z0, x1, y1, z1] =
      std::make_tuple(pmin.x()[0], pmin.y()[0], pmin.z()[0], pmax.x()[0],
                      pmax.y()[0], pmax.z()[0]);
  auto vertices = Vec3fD{FloatD{x0, x0, x0, x0, x1, x1, x1, x1},
                         FloatD{y0, y0, y1, y1, y0, y0, y1, y1},
                         FloatD{z0, z1, z0, z1, z0, z1, z0, z1}};
  auto face_indices = Vec3iC{IntC{0, 3, 4, 7, 0, 5, 2, 7, 4, 2, 5, 3},
                             IntC{1, 2, 5, 6, 4, 1, 6, 3, 0, 6, 1, 7},
                             IntC{2, 1, 6, 5, 1, 4, 3, 6, 6, 0, 7, 1}};

  auto u = Real{0.25};
  auto v = Real{1.0 / 3.0};
  auto uvs = Vec2fC{
      FloatC{u, 2 * u, 0, u, 2 * u, 3 * u, 4 * u, 0, u, 2 * u, 3 * u, 4 * u, u,
             2 * u},
      FloatC{0, 0, v, v, v, v, v, 2 * v, 2 * v, 2 * v, 2 * v, 2 * v, 1, 1},
  };
  auto uv_indices = Vec3iC{IntC{2, 8, 5, 9, 0, 4, 12, 9, 5, 11, 4, 8},
                           IntC{3, 7, 4, 10, 1, 3, 13, 8, 6, 10, 3, 9},
                           IntC{7, 3, 10, 4, 3, 1, 8, 13, 10, 6, 9, 3}};

  auto tex = std::shared_ptr<Texture3f>{m_cube_map_tex};
  auto mtrl = std::make_shared<DiffuseLightMaterial>(tex);
  m_mesh = std::make_shared<TriangleMesh>(vertices, uvs, face_indices,
                                          uv_indices, mtrl);

  // get triangle point sampler
  m_triangle_sampler =
      std::make_shared<AliasSampler>(enoki::detach(m_mesh->triangles().area));

  // compute power, TODO: get accurate power
  m_power = REAL_4PI;
}

template <bool ad>
Spectrum<ad> EnvLight::eval_(const Intersection<ad> &its,
                             const Mask<ad> &valid) const {
  auto new_valid = valid && its.valid;
  auto emit = its.material->emit(its, new_valid);
  // emit *= its.squared_dist / enoki::dot(its.normal, -its.wi);
  return emit;
}

template <bool ad>
SampledLightPoint<ad>
EnvLight::sample_light_point_(const Vec2f<ad> &sampler_,
                              const Mask<ad> &valid) const {
  auto sampler = sampler_;
  auto [idx, _] = m_triangle_sampler->sample_reuse<ad>(sampler.x(), valid);

  auto &triangles = m_mesh->triangles();
  auto sampled_triangles = Triangle<ad>{};
  if constexpr (ad) {
    sampled_triangles = enoki::gather<TriangleD>(triangles, idx, valid);
  } else {
    sampled_triangles =
        enoki::gather<TriangleC>(enoki::detach(triangles), idx, valid);
  }

  // p = sqrt(v)*(1-u) * A + u*sqrt(v) * B + (1-sqrt(v)) * C
  //   = A + u*sqrt(v)*(B-A) + (1-sqrt(v))*(C-A)
  auto &randu = sampler.x();
  auto &randv = sampler.y();
  auto sqrt_v = enoki::safe_sqrt(randv);
  auto u = randu * sqrt_v;
  auto v = 1 - sqrt_v;

  auto rst = SampledLightPoint<ad>{};
  rst.point = sampled_triangles.p0 +
              (sampled_triangles.p1 - sampled_triangles.p0) * u +
              (sampled_triangles.p2 - sampled_triangles.p0) * v;
  rst.pdf = enoki::rcp(enoki::detach(m_mesh->area()));
  return rst;
}

template <bool ad>
Float<ad> EnvLight::pdf_sampled_light_point_(const Vec3f<ad> &point,
                                             const Mask<ad> &valid) const {
  return enoki::select(valid, enoki::rcp(enoki::detach(m_mesh->area())), 0);
}

SpectrumC EnvLight::eval(const IntersectionC &its, const MaskC &valid) const {
  return eval_<false>(its, valid);
}
SpectrumD EnvLight::eval(const IntersectionD &its, const MaskD &valid) const {
  return eval_<true>(its, valid);
}

SampledLightPointC EnvLight::sample_light_point(const Vec2fC &sampler,
                                                const MaskC &valid) const {
  return sample_light_point_<false>(sampler, valid);
}
SampledLightPointD EnvLight::sample_light_point(const Vec2fD &sampler,
                                                const MaskD &valid) const {
  return sample_light_point_<true>(sampler, valid);
}

FloatC EnvLight::pdf_sampled_light_point(const Vec3fC &point,
                                         const MaskC &valid) const {
  return pdf_sampled_light_point_<false>(point, valid);
}
FloatD EnvLight::pdf_sampled_light_point(const Vec3fD &point,
                                         const MaskD &valid) const {
  return pdf_sampled_light_point_<true>(point, valid);
}

//////////////////// sh9 light /////////////////////
SH9Light::SH9Light(const Vec3fD &sh9_coeff)
    : Light{LightType::spherical_harmonics},
      m_sh9_coeff{
          enoki::gather<Vec3fD>(sh9_coeff, IntD{0}),
          enoki::gather<Vec3fD>(sh9_coeff, IntD{1}),
          enoki::gather<Vec3fD>(sh9_coeff, IntD{2}),
          enoki::gather<Vec3fD>(sh9_coeff, IntD{3}),
          enoki::gather<Vec3fD>(sh9_coeff, IntD{4}),
          enoki::gather<Vec3fD>(sh9_coeff, IntD{5}),
          enoki::gather<Vec3fD>(sh9_coeff, IntD{6}),
          enoki::gather<Vec3fD>(sh9_coeff, IntD{7}),
          enoki::gather<Vec3fD>(sh9_coeff, IntD{8}),
      } {}

void SH9Light::set_aabb(const Vec3fC &pmin, const Vec3fC &pmax) {
  // construct mesh
  auto [x0, y0, z0, x1, y1, z1] =
      std::make_tuple(pmin.x()[0], pmin.y()[0], pmin.z()[0], pmax.x()[0],
                      pmax.y()[0], pmax.z()[0]);
  auto vertices = Vec3fD{FloatD{x0, x0, x0, x0, x1, x1, x1, x1},
                         FloatD{y0, y0, y1, y1, y0, y0, y1, y1},
                         FloatD{z0, z1, z0, z1, z0, z1, z0, z1}};
  auto face_indices = Vec3iC{IntC{0, 3, 4, 7, 0, 5, 2, 7, 4, 2, 5, 3},
                             IntC{1, 2, 5, 6, 4, 1, 6, 3, 0, 6, 1, 7},
                             IntC{2, 1, 6, 5, 1, 4, 3, 6, 6, 0, 7, 1}};

  auto uvs = Vec2fC{};
  auto uv_indices = Vec3iC{};

  auto mtrl = std::make_shared<DiffuseLightMaterial>(Vec3fD{1});
  m_mesh = std::make_shared<TriangleMesh>(vertices, uvs, face_indices,
                                          uv_indices, mtrl);

  // get triangle point sampler
  m_triangle_sampler =
      std::make_shared<AliasSampler>(enoki::detach(m_mesh->triangles().area));

  // compute power, TODO: get accurate power
  m_power = REAL_4PI;
}

template <bool ad>
Spectrum<ad> SH9Light::eval_impl_(const Vec3f<ad> &dir,
                                  const Mask<ad> &valid) const {
  auto &x = dir.x();
  auto &y = dir.y();
  auto &z = dir.z();
  auto xx = x * x;
  auto yy = y * y;
  auto zz = z * z;
  auto xy = x * y;
  auto yz = y * z;
  auto zx = z * x;
  auto Y = Coeff<ad>{
      std::sqrt(REAL_INV_4PI),                                      // Y(0, 0)
      std::sqrt(Real{0.75} * REAL_INV_PI) * y,                      // Y(1,-1)
      std::sqrt(Real{0.75} * REAL_INV_PI) * z,                      // Y(1, 0)
      std::sqrt(Real{0.75} * REAL_INV_PI) * x,                      // Y(1,1)
      std::sqrt(Real{15 / 4.0} * REAL_INV_PI) * xy,                 // Y(2, -2)
      std::sqrt(Real{15 / 4.0} * REAL_INV_PI) * yz,                 // Y(2,-1)
      std::sqrt(Real{5 / 16.0} * REAL_INV_PI) * (2 * zz - xx - yy), // Y(2, 0)
      std::sqrt(Real{15 / 4.0} * REAL_INV_PI) * zx,                 // Y(2,1)
      std::sqrt(Real{15 / 16.0} * REAL_INV_PI) * (xx - yy),         // Y(2, 2)
  };

  auto C = Coeff3f<ad>{};
  if constexpr (ad) {
    C = m_sh9_coeff;
  } else {
    C = enoki::detach(m_sh9_coeff);
  }

  return enoki::select(valid, enoki::hsum(C * Y), 0);
}

template <bool ad>
Spectrum<ad> SH9Light::eval_(const Intersection<ad> &its,
                             const Mask<ad> &valid) const {
  auto new_valid = valid && its.valid;
  return eval_impl_<ad>(its.wi, new_valid);
  // return eval_impl_<ad>(its.wi, new_valid) * its.squared_dist /
  //        enoki::dot(its.normal, -its.wi);
}

template <bool ad>
SampledLightPoint<ad>
SH9Light::sample_light_point_(const Vec2f<ad> &sampler_,
                              const Mask<ad> &valid) const {
  auto sampler = sampler_;
  auto [idx, _] = m_triangle_sampler->sample_reuse<ad>(sampler.x(), valid);

  auto &triangles = m_mesh->triangles();
  auto sampled_triangles = Triangle<ad>{};
  if constexpr (ad) {
    sampled_triangles = enoki::gather<TriangleD>(triangles, idx, valid);
  } else {
    sampled_triangles =
        enoki::gather<TriangleC>(enoki::detach(triangles), idx, valid);
  }

  // p = sqrt(v)*(1-u) * A + u*sqrt(v) * B + (1-sqrt(v)) * C
  //   = A + u*sqrt(v)*(B-A) + (1-sqrt(v))*(C-A)
  auto &randu = sampler.x();
  auto &randv = sampler.y();
  auto sqrt_v = enoki::safe_sqrt(randv);
  auto u = randu * sqrt_v;
  auto v = 1 - sqrt_v;

  auto rst = SampledLightPoint<ad>{};
  rst.point = sampled_triangles.p0 +
              (sampled_triangles.p1 - sampled_triangles.p0) * u +
              (sampled_triangles.p2 - sampled_triangles.p0) * v;
  rst.pdf = enoki::rcp(enoki::detach(m_mesh->area()));
  return rst;
}

template <bool ad>
Float<ad> SH9Light::pdf_sampled_light_point_(const Vec3f<ad> &point,
                                             const Mask<ad> &valid) const {
  return enoki::select(valid, 1 / enoki::detach(m_mesh->area()), 0);
}

SpectrumC SH9Light::eval(const IntersectionC &its, const MaskC &valid) const {
  return eval_<false>(its, valid);
}
SpectrumD SH9Light::eval(const IntersectionD &its, const MaskD &valid) const {
  return eval_<true>(its, valid);
}

SampledLightPointC SH9Light::sample_light_point(const Vec2fC &sampler,
                                                const MaskC &valid) const {
  return sample_light_point_<false>(sampler, valid);
}
SampledLightPointD SH9Light::sample_light_point(const Vec2fD &sampler,
                                                const MaskD &valid) const {
  return sample_light_point_<true>(sampler, valid);
}

FloatC SH9Light::pdf_sampled_light_point(const Vec3fC &point,
                                         const MaskC &valid) const {
  return pdf_sampled_light_point_<false>(point, valid);
}
FloatD SH9Light::pdf_sampled_light_point(const Vec3fD &point,
                                         const MaskD &valid) const {
  return pdf_sampled_light_point_<true>(point, valid);
}

// explicit SH9Light(const Vec3fD& sh9_coeff, uint32_t tex_size = 16)
//     : Light{LightType::spherical_armonics} {
//   // reconstruct envmap with resolution tex_size*tex_size from [-1,1]^2
//   auto c = 0;
//   auto d = 2;
//   auto [x0, y0, z0, x1, y1, z1] = std::make_tuple(-1, -1, -1, 1, 1, 1);
//   auto [dx, dy, dz] = std::make_tuple(2, 2, 2);
//   auto cube_map_data = enoki::zero<Vec3fD>(m_tex_size * 4 * 3);
//   auto idx = enoki::empty<IntD>(m_tex_size * 6);
//   auto pos = enoki::empty<Vec3fD>(m_tex_size * 6);

//   auto szsz = m_tex_size * m_tex_size;
//   auto tmp = enoki::arange<IntD>(szsz);
//   auto row = tmp / m_tex_size;
//   auto col = tmp % m_tex_size;
//   auto stride = m_tex_size * 4;
//   auto base = row * stride + col;

//   auto offset = 0u;
//   auto grid = (enoki::meshgrid<FloatD>(enoki::arange<FloatD>(m_tex_size),
//                                        enoki::arange<FloatD>(m_tex_size)) +
//                Real{0.5}) /
//               m_tex_size;
//   // bottom
//   enoki::scatter(idx, base + m_tex_size, tmp + offset);
//   auto xz = grid * Vec2fD{dx, dz} + Vec2fD{x0, z0};
//   auto pos_bottom = Vec3fD{xz.x(), y0, xz.y()};
//   enoki::scatter(pos, pos_bottom, tmp + offset);
//   offset += szsz;
//   // right
//   enoki::scatter(idx, base + m_tex_size * stride, tmp + offset);
//   auto zy = grid * Vec2fD{dz, dy} + Vec2fD{z0, y0};
//   auto pos_right = Vec3fD{x0, zy.y(), zy.x()};
//   enoki::scatter(pos, pos_right, tmp + offset);
//   offset += szsz;
//   // front
//   enoki::scatter(idx, base + (m_tex_size * stride + m_tex_size),
//                  tmp + offset);
//   auto xy = grid * Vec2fD{dx, dy} + Vec2fD{x0, y0};
//   auto pos_front = Vec3fD{xy.x(), xy.y(), z1};
//   enoki::scatter(pos, pos_front, tmp + offset);
//   offset += szsz;
//   // left
//   enoki::scatter(idx, base + (m_tex_size * stride + m_tex_size * 2),
//                  tmp + offset);
//   auto zy_ = grid * Vec2fD{-dz, dy} + Vec2fD{z1, y0};
//   auto pos_left = Vec3fD{x1, zy_.y(), zy_.x()};
//   enoki::scatter(pos, pos_left, tmp + offset);
//   offset += szsz;
//   // back
//   enoki::scatter(idx, base + (m_tex_size * stride + m_tex_size * 3),
//                  tmp + offset);
//   auto xy_ = grid * Vec2fD{-dx, dy} + Vec2fD{x1, y0};
//   auto pos_back = Vec3fD{xy_.x(), xy_.y(), z0};
//   enoki::scatter(pos, pos_back, tmp + offset);
//   offset += szsz;
//   // top
//   enoki::scatter(idx, base + (2 * m_tex_size * stride + m_tex_size),
//                  tmp + offset);
//   auto xz_ = grid * Vec2fD{dx, -dz} + Vec2fD{x0, z1};
//   auto pos_top = Vec3fD{xz_.x(), y1, xz_.y()};
//   enoki::scatter(pos, pos_top, tmp + offset);

//   pos -= c;
//   pos = enoki::normalize(pos);
//   auto xx = pos.x() * pos.x();
//   auto yy = pos.y() * pos.y();
//   auto zz = pos.z() * pos.z();
//   auto y00 = FloatD{Real{0.5} * std::sqrt(REAL_INV_PI)};
//   auto y11_ = std::sqrt(Real{0.75} * REAL_INV_PI) * pos.y();
//   auto y10 = std::sqrt(Real{0.75} * REAL_INV_PI) * pos.z();
//   auto y11 = std::sqrt(Real{0.75} * REAL_INV_PI) * pos.x();
//   auto y22_ = std::sqrt(Real{15 / 4.0} * REAL_INV_PI) * (pos.x() *
//   pos.y()); auto y21_ = std::sqrt(Real{15 / 4.0} * REAL_INV_PI) * (pos.y()
//   * pos.z()); auto y20 = std::sqrt(Real{5 / 16.0} * REAL_INV_PI) * (2 * zz
//   - xx - yy); auto y21 = std::sqrt(Real{15 / 4.0} * REAL_INV_PI) * (pos.z()
//   * pos.x()); auto y22 = std::sqrt(Real{15 / 16.0} * REAL_INV_PI) * (xx -
//   yy);

//   auto y = std::vector<FloatD*>{&y00,  &y11_, &y10, &y11, &y22_,
//                                 &y21_, &y20,  &y21, &y22};
//   auto tex = Vec3fD{0};
//   for (auto i = 0u; i < 9u; ++i) {
//     auto ci = enoki::gather<Vec3fD>(m_emit, IntD{i});
//     auto& yi = *(y[i]);
//     tex += ci * yi;
//   }
//   enoki::scatter(cube_map_data, tex, idx);
//   auto cube_map_tex = std::make_shared<UVTexture3f>(
//       3 * m_tex_size, 4 * m_tex_size, cube_map_data);
//   auto mtrl = std::make_shared<DiffuseLightMaterial>(
//       std::shared_ptr<Texture3f>{cube_map_tex});

//   auto vertices = Vec3fD{FloatD{x0, x0, x0, x0, x1, x1, x1, x1},
//                          FloatD{y0, y0, y1, y1, y0, y0, y1, y1},
//                          FloatD{z0, z1, z0, z1, z0, z1, z0, z1}};
//   auto face_indices = Vec3iC{IntC{0, 3, 4, 7, 0, 5, 2, 7, 0, 2, 1, 3},
//                              IntC{1, 2, 5, 6, 4, 1, 6, 3, 4, 6, 5, 7},
//                              IntC{2, 1, 6, 5, 1, 4, 3, 6, 6, 4, 7, 5}};

//   auto u = Real{0.25};
//   auto v = Real{1 / 3.0};
//   auto uvs = Vec2fD{
//       FloatD{u, 2 * u, 0, u, 2 * u, 3 * u, 4 * u, 0, u, 2 * u, 3 * u, 4 *
//       u,
//              u, 2 * u},
//       FloatD{0, 0, v, v, v, v, v, 2 * v, 2 * v, 2 * v, 2 * v, 2 * v, 1, 1},
//   };
//   auto uv_indices = Vec3iC{IntC{2, 8, 5, 9, 0, 4, 12, 9, 6, 11, 3, 8},
//                            IntC{3, 7, 4, 10, 1, 3, 13, 8, 5, 10, 4, 9},
//                            IntC{7, 3, 10, 4, 3, 1, 8, 13, 10, 5, 9, 3}};
// }