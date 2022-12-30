#include "src/path_integrator.h"

template <bool ad>
Float<ad> PathIntegrator::mis_weight(const Float<ad> &w1, const Float<ad> &w2,
                                     uint32_t n1, uint32_t n2) const {
  auto sqr_w1 = enoki::sqr(n1 * w1);
  auto sqr_w2 = enoki::sqr(n2 * w2);
  return sqr_w1 / (sqr_w1 + sqr_w2);
}

template <bool ad>
Vec3f<ad> PathIntegrator::hit_triangle(const Vec3f<ad> &o, const Vec3f<ad> &d,
                                       const Vec3f<ad> &p0, const Vec3f<ad> &e1,
                                       const Vec3f<ad> &e2) const {
  auto &E1 = e1;
  auto &E2 = e2;
  auto T = o - p0;
  auto &D = d;

  auto P = enoki::cross(D, E2);
  auto Q = enoki::cross(T, E1);
  // NOTE: no clamp called will cause grad NAN.
  auto inv_PE1 =
      enoki::rcp(enoki::clamp(enoki::dot(P, E1), REAL_EPS, 1 - REAL_EPS));
  // auto t = enoki::dot(Q, E2) * inv_PE1;
  auto u = enoki::dot(P, T) * inv_PE1;
  auto v = enoki::dot(Q, D) * inv_PE1;
  return p0 + u * E1 + v * E2;
}

template <bool ad>
Spectrum<ad>
PathIntegrator::boundary_term(const Vec3f<ad> &xs, const Vec3f<ad> &xd,
                              const Vec3f<ad> &xb, const Vec3f<ad> &ns,
                              const Vec3f<ad> &nd, const Edge<ad> &edge) const {
  auto xs_ = enoki::detach(xs);
  auto xd_ = enoki::detach(xd);
  auto xb_ = enoki::detach(xb);
  auto ns_ = enoki::detach(ns);
  auto nd_ = enoki::detach(nd);

  auto dir = enoki::normalize(xb - xs_);
  auto dir_ = enoki::detach(dir);
  // auto edge_ = enoki::detach(edge);

  auto edge_vec_ =
      enoki::normalize(enoki::detach(edge.p1) - enoki::detach(edge.p0));
  auto mask_reverse = enoki::dot(enoki::cross(dir_, edge_vec_),
                                 enoki::detach(edge.out_direction)) < 0;
  edge_vec_[mask_reverse] *= -1;

  auto n_plane_ = enoki::cross(dir_, edge_vec_);
  auto edge_proj_ = enoki::cross(nd_, n_plane_);

  auto cos_theta_s = enoki::abs_dot(ns_, dir_);
  auto cos_theta_b = enoki::dot(edge_vec_, dir_);
  auto cos_theta_d = enoki::dot(edge_proj_, dir_);
  auto sin_theta_b = enoki::safe_sqrt(1 - cos_theta_b * cos_theta_b);
  auto sin_theta_d = enoki::safe_sqrt(1 - cos_theta_d * cos_theta_d);

  auto len_sb = enoki::norm(xb_ - xs_);
  auto len_sd = enoki::norm(xd_ - xs_);
  auto Jb = len_sd * len_sb * sin_theta_b / (sin_theta_d * cos_theta_s);

  auto delta_G = cos_theta_s * enoki::dot(nd_, -dir_) / (len_sd * len_sd);

  auto fb = Vec3f<ad>{0};
  if constexpr (ad) {
    auto n_edge_proj_ = enoki::cross(nd_, edge_proj_);
    auto xd_hit = hit_triangle<ad>(xs_, dir, xd_, edge_proj_, n_edge_proj_);
    xd_hit -= enoki::detach(xd_hit);
    fb = xd_hit * (delta_G * n_edge_proj_);
  } else {
    fb = delta_G;
  }

  return fb * Jb;
}

template <bool ad>
Spectrum<ad> PathIntegrator::trace2light(
    const Scene &scene, const Sampler &sampler, const Intersection<ad> &its0,
    const Mask<ad> &valid_, uint32_t light_samples, uint32_t bsdf_samples,
    uint32_t max_depth, bool enable_light_visable) const {
  auto T = Spectrum<ad>{1};
  auto L = Spectrum<ad>{0};

  auto its = its0;
  auto valid = valid_ && its0.valid;
  // auto itss = std::vector<Intersection<ad>>(max_depth);
  for (auto depth = 0u; depth < max_depth; ++depth) {
    { // hit light
      auto hit_light = valid && its.is_light;
      if (enable_light_visable && depth == 0) {
        auto emit = its.light->eval(its, hit_light);
        L[hit_light] += T * emit;
      }
      valid &= !hit_light;
    }

    { // direct illumination
      // sample light
      for (auto i = 0u; i < light_samples; ++i) {
        auto randuv = sampler.next_nd<2, ad>(valid);
        auto [light, pdf_choose_light_] =
            scene.sample_light_reuse<ad>(randuv.x(), valid);
        auto [light_point, pdf_light_] =
            light->sample_light_point(randuv, valid);
        pdf_light_ *= pdf_choose_light_;
        auto pdf_light = enoki::detach(pdf_light_);
        auto ray =
            Ray<ad>{its.point, enoki::normalize(light_point - its.point)};
        auto its_light = scene.hit(ray, valid);
        auto is_hit =
            valid && its_light.valid && its_light.is_light &&
            enoki::squared_norm(its_light.point - light_point) < REAL_EPS;
        auto emit = light->eval(its_light, is_hit);
        auto bsdf = its.material->eval(its, ray.direction, is_hit);
        auto tmp = enoki::dot(its_light.normal, -ray.direction) /
                   its_light.squared_dist;
        auto G = enoki::dot(its.normal, ray.direction) * tmp;
        auto value = emit * G * bsdf * its_light.jacobi / pdf_light;
        auto weight = FloatC{Real{1} / light_samples};
        if (bsdf_samples > 0) {
          auto pdf_bsdf_ = its.material->pdf(its, ray.direction, is_hit) * tmp;
          auto pdf_bsdf = enoki::detach(pdf_bsdf_);
          weight *= mis_weight<false>(pdf_light, pdf_bsdf, light_samples,
                                      bsdf_samples);
        }
        L[is_hit] += T * weight * value;
      }

      // sample bsdf
      for (auto i = 0u; i < bsdf_samples; ++i) {
        auto randuv = sampler.next_nd<2, ad>(valid);
        auto sampled_bsdf = its.material->sample(its, randuv, valid);
        auto ray = Ray<ad>{its.point, sampled_bsdf.wo};
        auto its_light = scene.hit(ray, valid);
        auto is_hit = valid && its_light.valid && its_light.is_light;
        auto emit = its_light.light->eval(its_light, is_hit);
        auto &bsdf = sampled_bsdf.bsdf;
        auto tmp = enoki::dot(its_light.normal, -ray.direction) /
                   its_light.squared_dist;
        auto G = enoki::dot(its.normal, ray.direction) * tmp;
        auto pdf_bsdf_ = sampled_bsdf.pdf * tmp;
        auto pdf_bsdf = enoki::detach(pdf_bsdf_);
        auto value = emit * G * bsdf * its_light.jacobi / pdf_bsdf;

        auto weight = FloatC{Real{1} / bsdf_samples};
        if (light_samples > 0) {
          auto pdf_light_ =
              scene.pdf_sampled_light<ad>(its_light.light, is_hit) *
              its_light.light->pdf_sampled_light_point(its_light.point, is_hit);
          auto pdf_light = enoki::detach(pdf_light_);

          weight *= mis_weight<false>(pdf_bsdf, pdf_light, bsdf_samples,
                                      light_samples);
        }
        L[is_hit] += T * weight * value;
      }
    }

    if (depth < max_depth) { // indirect illumination
      auto randuv = sampler.next_nd<2, ad>(valid);
      auto sampled_bsdf = its.material->sample(its, randuv, valid);
      auto ray = Ray<ad>{its.point, sampled_bsdf.wo};
      // auto& its_next = itss[depth];
      auto its_next = scene.hit(ray, valid);
      valid &= sampled_bsdf.valid && its_next.valid;
      auto &bsdf = sampled_bsdf.bsdf;
      auto tmp =
          enoki::dot(its_next.normal, -ray.direction) / its_next.squared_dist;
      auto G = enoki::dot(its.normal, ray.direction) * tmp;
      auto pdf_bsdf_ = sampled_bsdf.pdf * tmp;
      T[valid] *= bsdf * its_next.jacobi * G / enoki::detach(pdf_bsdf_);
      // NOTE: this std::move is very important, thus code will leak of mem
      // without it.
      its = std::move(its_next);
    } else {
      break;
    }
  }

  L[!enoki::isfinite(enoki::squared_norm(L))] = 0;
  return L;
}

template <bool ad>
void PathIntegrator::render_primary_boundary(const Scene &scene,
                                             const BaseCamera &camera,
                                             Film<ad> &result) const {
  auto hyper_cube_sampler = preprocess_primary_boundary(scene, camera);
  auto sampler = camera.gen_sampler(m_cfg.spp_primary_edge);
  auto sampler1 = sampler.next_nd<1, ad>();
  auto pdf = hyper_cube_sampler.sample_reuse<ad>(sampler1);
  auto [raster_coord, value, valid] =
      eval_primary_boundary<ad>(scene, camera, sampler, sampler1[0]);
  valid &= enoki::isfinite(enoki::squared_norm(value));
  result.add(value/pdf, raster_coord, valid, m_cfg.spp_primary_edge);
}

template <bool ad>
std::tuple<Vec2i<ad>, Spectrum<ad>, Mask<ad>>
PathIntegrator::eval_primary_boundary(const Scene &scene,
                                      const BaseCamera &camera,
                                      const Sampler &sampler,
                                      const Float<ad> &sampler1) const {
  auto [edge, edge_point, pdf_edge_point] =
      scene.sample_edge_point<ad>(sampler1);
      
  auto [ray, raster_coord, valid] =
      camera.gen_camera_ray_with_point(edge_point);
  auto &ray_direction = ray.direction;
  auto &ray_origin = ray.origin;
  auto edge_dir = edge.p1 - edge.p0;
  auto edge_normal_ =
      enoki::cross(ray_direction, enoki::normalize(enoki::detach(edge_dir)));
  auto edge_normal = enoki::detach(edge_normal_);

  auto o = enoki::detach(ray_origin);
  auto d = enoki::detach(ray_direction);
  auto its_a =
      scene.hit(RayC{o, d - REAL_EPS * edge_normal}, enoki::detach(valid));
  auto its_b =
      scene.hit(RayC{o, d + REAL_EPS * edge_normal}, enoki::detach(valid));

  // TODO: one is valid and another is not valid may also can be computed
  valid &= its_a.valid && its_b.valid &&
           (enoki::norm(its_a.point-enoki::detach(edge_point)) < 10 * REAL_EPS ||
            enoki::norm(its_b.point-enoki::detach(edge_point)) < 10 * REAL_EPS);

  auto delta_L =
      trace2light<false>(scene, sampler, its_a, enoki::detach(valid),
                         m_cfg.mis_light_samples, 1, m_cfg.max_bounce, false) -
      trace2light<false>(scene, sampler, its_b, enoki::detach(valid),
                         m_cfg.mis_light_samples, 1, m_cfg.max_bounce, false);

  auto value = Spectrum<ad>{};
  if constexpr (ad) {
    // TODO: to understand why shoud times [1, 1, -1]
    auto x_dot_n =
        enoki::dot(edge_point - enoki::detach(edge_point), edge_normal) +
        enoki::dot(ray_origin - enoki::detach(ray_origin),
                   edge_normal * Vec3fC{1, 1, -1});
    value = x_dot_n * Spectrum<ad>{delta_L} / enoki::detach(pdf_edge_point);
  } else {
    value = delta_L / pdf_edge_point;
  }
  valid &= enoki::isfinite(enoki::squared_norm(value));
  return {raster_coord, value, valid};
}

HyperCubeSampler1
PathIntegrator::preprocess_primary_boundary(const Scene &scene,
                                            const BaseCamera &camera) const {
  auto cube_sampler =
      HyperCubeSampler1{IntC{m_cfg.primary_edge_hypercube_resolution}};
  auto cells_cnt = cube_sampler.cells_cnt();
  auto idx = enoki::arange<IntC>(cells_cnt);
  auto sample_base = enoki::gather<VecC<Integer, 1>>(cube_sampler.cells(), idx);
  auto sampler = Sampler{cells_cnt};
  auto rst = enoki::zero<FloatC>(cells_cnt);
  for (auto i = 0u; i < m_cfg.primary_edge_preprocess_rounds; ++i) {
    auto sampler1 =
        (sample_base + sampler.next_nd<1, false>()) * cube_sampler.unit();
    auto [_, value, valid] =
        eval_primary_boundary<false>(scene, camera, sampler, sampler1[0]);
    enoki::scatter_add(rst, enoki::norm(value), idx, valid);
  }
  cube_sampler.set_mass<false>(rst);
  return cube_sampler;
}

template <bool ad>
void PathIntegrator::render_direct_boundary(const Scene &scene,
                                            const BaseCamera &camera,
                                            Film<ad> &result) const {
  auto hyper_cube_sampler = preprocess_direct_boundary(scene, camera);
  auto sampler = camera.gen_sampler(m_cfg.spp_secondary_edge);
  auto sampler3 = sampler.next_nd<3, ad>();
  auto pdf = hyper_cube_sampler.sample_reuse<ad>(sampler3);
  auto [raster_coord, value, valid] =
      eval_direct_boundary<ad>(scene, camera, sampler3);
  result.add(value/pdf, raster_coord, valid, m_cfg.spp_secondary_edge);
}

template <bool ad>
std::tuple<Vec2i<ad>, Spectrum<ad>, Mask<ad>>
PathIntegrator::eval_direct_boundary(const Scene &scene, const BaseCamera &camera,
                                     const Vec3f<ad> &sampler_) const {
  auto sampler = sampler_;
  // sample edge and light
  auto [edge, edge_point, pdf_edge_point] =
      scene.sample_edge_point<ad>(sampler.x());
  auto [light, pdf_choose_light] = scene.sample_light_reuse<ad>(sampler.y());
  auto [light_point, pdf_light] =
      light->sample_light_point(enoki::tail<2>(sampler));
  pdf_light *= pdf_choose_light;

  auto ray_direction = enoki::normalize(edge_point - light_point);
  auto valid = enoki::dot(edge.n_left, ray_direction) *
                       enoki::dot(edge.n_right, ray_direction) <
                   0 &&
               enoki::norm(edge.n_left - edge.n_right) > REAL_EPS;
  auto its_s0 = scene.hit(Ray<ad>{edge_point, -ray_direction}, valid);
  valid &= its_s0.valid && its_s0.is_light &&
           enoki::squared_norm(its_s0.point - light_point) < REAL_EPS;
  auto its_d0 = scene.hit(Ray<ad>{edge_point, ray_direction}, valid);
  valid &= its_d0.valid && !its_d0.is_light;

  auto [cam_ray, raster_coord, valid_point] = camera.gen_camera_ray_with_point(its_d0.point, valid);
  valid &= valid_point;
  auto its2d0 =
      scene.hit(cam_ray, valid);
  valid &= its2d0.valid && !its2d0.is_light &&
           enoki::squared_norm(its2d0.point - its_d0.point) < REAL_EPS;

  auto pdf_xw = its_s0.squared_dist * pdf_edge_point * pdf_light /
                enoki::dot(its_s0.normal, ray_direction);

  auto Tb = boundary_term<ad>(its_s0.point, its_d0.point, edge_point,
                              its_s0.normal, its_d0.normal, edge) /
            pdf_xw;
  auto Ts = its_s0.light->eval(its_s0, valid);
  auto Td = its_d0.material->eval(its_d0, -cam_ray.direction, valid);
  auto value = Tb * enoki::detach(Ts) * enoki::detach(Td);
  valid &= enoki::isfinite(enoki::squared_norm(value));
  return {raster_coord, value, valid};
}

HyperCubeSampler3
PathIntegrator::preprocess_direct_boundary(const Scene &scene,
                                           const BaseCamera &camera) const {
  auto &reso = m_cfg.secondary_edge_hypercube_resolution;
  auto cube_sampler =
      HyperCubeSampler3{Vec3iC{reso.at(0), reso.at(1), reso.at(2)}};
  auto cells_cnt = cube_sampler.cells_cnt();
  auto idx = enoki::arange<IntC>(cells_cnt);
  auto sample_base = enoki::gather<Vec3iC>(cube_sampler.cells(), idx);
  auto sampler = Sampler{cells_cnt};
  auto rst = enoki::zero<FloatC>(cells_cnt);
  for (auto i = 0u; i < m_cfg.secondary_edge_preprocess_rounds; ++i) {
    auto sampler3 =
        (sample_base + sampler.next_nd<3, false>()) * cube_sampler.unit();
    auto [_, value, valid] =
        eval_direct_boundary<false>(scene, camera, sampler3);
    value[value < REAL_EPS] = REAL_EPS;
    auto val = enoki::norm(value);
    enoki::scatter_add(rst, val, idx, valid);
  }
  cube_sampler.set_mass<false>(rst);
  return cube_sampler;
}

template <bool ad>
void PathIntegrator::render_interior(const Scene &scene,
                                     const BaseCamera &camera,
                                     Film<ad> &result) const {
  auto sampler = camera.gen_sampler(m_cfg.spp_interior);
  auto sample = sampler.next_nd<2, ad>();
  auto [ray, raster_coord] = camera.gen_camera_ray(sample);
  auto its = scene.hit(ray);
  auto valid = its.valid;

  auto spectrum =
      trace2light<ad>(scene, sampler, its, valid, m_cfg.mis_light_samples,
                      m_cfg.mis_bsdf_samples, m_cfg.max_bounce,
                      m_cfg.enable_light_visable) *
      its.jacobi;

  result.add(spectrum, raster_coord, valid, m_cfg.spp_interior);
}

template <bool ad>
void PathIntegrator::render_boundary(const Scene &scene, const BaseCamera &camera,
                                     Film<ad> &result) const {
  render_primary_boundary<ad>(scene, camera, result);
  render_direct_boundary<ad>(scene, camera, result);
}

template <bool ad>
std::vector<Spectrum<ad>> PathIntegrator::render_(const Scene &scene) const {
  auto cameras = scene.cameras();
  auto rsts = std::vector<Spectrum<ad>>{};
  for (auto camera : cameras) {
    auto film = Film<ad>{camera->height(), camera->width()};

    for (auto n = 0u; n < m_cfg.n_pass; ++n) {
      // interior render
      render_interior<ad>(scene, *camera, film);
      // boundary render
      if constexpr (ad) {
        if (scene.need_optimize_boundary()) {
          render_boundary<true>(scene, *camera, film);
        }
      }
    }
    if (m_cfg.n_pass > 1) {
      film.div(m_cfg.n_pass);
    }

    rsts.emplace_back(film.data());
  }
  enoki::cuda_eval();
  return rsts;
}

std::vector<SpectrumC> PathIntegrator::renderC(const Scene &scene) const {
  return render_<false>(scene);
}

std::vector<SpectrumD> PathIntegrator::renderD(const Scene &scene) const {
  return render_<true>(scene);
}
