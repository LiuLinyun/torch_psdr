#pragma once

#include "src/camera.h"
#include "src/integrator.h"

class PathIntegrator : public Integrator {
public:
  struct Config {
    uint32_t n_pass{1};
    uint32_t spp_interior{1};
    bool enable_light_visable{false};
    uint32_t spp_primary_edge{1};
    uint32_t spp_secondary_edge{1};
    uint32_t max_bounce{2};
    uint32_t mis_light_samples{1};
    uint32_t mis_bsdf_samples{1};
    uint32_t primary_edge_preprocess_rounds{1};
    uint32_t secondary_edge_preprocess_rounds{1};
    uint32_t primary_edge_hypercube_resolution{10000};
    std::vector<uint32_t> secondary_edge_hypercube_resolution{{10000, 6, 6}};
  };

public:
  PathIntegrator() = default;
  explicit PathIntegrator(const Config &cfg) : m_cfg{cfg} {}
  virtual std::vector<SpectrumC> renderC(const Scene &scene) const override;
  virtual std::vector<SpectrumD> renderD(const Scene &scene) const override;

private:
  template <bool ad>
  std::vector<Spectrum<ad>> render_(const Scene &scene) const;

  template <bool ad>
  Spectrum<ad> trace2light(const Scene &scene, const Sampler &sampler,
                           const Intersection<ad> &its0, const Mask<ad> &valid_,
                           uint32_t light_samples, uint32_t bsdf_samples,
                           uint32_t max_depth, bool enable_light_visable) const;

  template <bool ad>
  void render_interior(const Scene &scene, const BaseCamera &camera,
                       Film<ad> &result) const;

  template <bool ad>
  void render_boundary(const Scene &scene, const BaseCamera &camera,
                       Film<ad> &result) const;

  template <bool ad>
  void render_primary_boundary(const Scene &scene, const BaseCamera &camera,
                               Film<ad> &result) const;
  template <bool ad>
  void render_direct_boundary(const Scene &scene, const BaseCamera &camera,
                              Film<ad> &result) const;

  template <bool ad>
  std::tuple<Vec2i<ad>, Spectrum<ad>, Mask<ad>>
  eval_primary_boundary(const Scene &scene, const BaseCamera &camera,
                        const Sampler &sampler,
                        const Float<ad> &sampler1) const;

  HyperCubeSampler1 preprocess_primary_boundary(const Scene &scene,
                                                const BaseCamera &camera) const;

  template <bool ad>
  std::tuple<Vec2i<ad>, Spectrum<ad>, Mask<ad>>
  eval_direct_boundary(const Scene &scene, const BaseCamera &camera,
                       const Vec3f<ad> &sampler) const;

  HyperCubeSampler3 preprocess_direct_boundary(const Scene &scene,
                                               const BaseCamera &camera) const;

  template <bool ad>
  Float<ad> mis_weight(const Float<ad> &w1, const Float<ad> &w2, uint32_t n1,
                       uint32_t n2) const;

  template <bool ad>
  Vec3f<ad> hit_triangle(const Vec3f<ad> &o, const Vec3f<ad> &d,
                         const Vec3f<ad> &p0, const Vec3f<ad> &e1,
                         const Vec3f<ad> &e2) const;

  template <bool ad>
  Spectrum<ad> boundary_term(const Vec3f<ad> &xs, const Vec3f<ad> &xd,
                             const Vec3f<ad> &xb, const Vec3f<ad> &ns,
                             const Vec3f<ad> &nd, const Edge<ad> &edge) const;

  Config m_cfg;
};