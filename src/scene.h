#pragma once

#include "src/camera.h"
#include "src/emitter.h"
#include "src/mesh.h"
#include "src/ray.h"
#include "src/scene_optix.h"
#include "src/types.h"

class Scene final {
public:
  Scene() = delete;
  Scene(const std::vector<std::shared_ptr<BaseCamera>> cameras,
        const std::vector<std::shared_ptr<TriangleMesh>> &objects,
        const std::vector<std::shared_ptr<Light>> &lights);

  const std::vector<std::shared_ptr<BaseCamera>> &cameras() const {
    return m_cameras;
  }

  template <bool ad>
  Intersection<ad> hit(const Ray<ad> &ray, const Mask<ad> &valid = true) const;

  template <bool ad>
  std::tuple<Type<Light *, ad>, Float<ad>>
  sample_light_reuse(Float<ad> &sampler, const Mask<ad> &valid = true) const;

  template <bool ad>
  Float<ad> pdf_sampled_light(const Type<Light *, ad> &light,
                              const Mask<ad> &valid = true) const;

  template <bool ad>
  std::tuple<Edge<ad>, Vec3f<ad>, Float<ad>>
  sample_edge_point(const Float<ad> &sampler,
                    const Mask<ad> &valid = true) const;

  template <bool ad>
  std::tuple<Type<Light *, ad>, Vec3f<ad>, Float<ad>>
  sample_direct_light_point(const Vec2f<ad> &sampler,
                            const Mask<ad> &valid = true) const;

  template <bool ad>
  std::tuple<Vec3f<ad>, Float<ad>, Mask<ad>>
  sample_object_edge_direction(const Sampler &sampler,
                               const Edge<ad> &sampled_edges,
                               const Mask<ad> &valid) const;

  template <bool ad>
  std::tuple<Float<ad>, Mask<ad>>
  sampled_object_edge_direction_pdf(const Edge<ad> &sampled_edges,
                                    const Vec3f<ad> &wo,
                                    const Mask<ad> &valid) const;

  bool need_optimize_boundary() const { return m_need_optimize_boundary; }

private:
  std::vector<std::shared_ptr<BaseCamera>> m_cameras;
  std::vector<std::shared_ptr<TriangleMesh>> m_meshes;
  std::vector<std::shared_ptr<Light>> m_emitters;

  TriangleD m_triangle;
  TypeD<Material *> m_material;
  TypeD<Light *> m_light;
  FloatD m_all_lights_power;
  std::shared_ptr<AliasSampler> m_light_idx_sampler;

  EdgeD m_object_edge;
  FloatD m_all_edge_length;
  std::shared_ptr<AliasSampler> m_edge_sampler;
  bool m_need_optimize_boundary;

  std::shared_ptr<SceneOptix> m_scene_optix;
};