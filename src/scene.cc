#include "src/scene.h"

Scene::Scene(const std::vector<std::shared_ptr<BaseCamera>> cameras,
             const std::vector<std::shared_ptr<TriangleMesh>> &meshes,
             const std::vector<std::shared_ptr<Light>> &emitters)
    : m_cameras{cameras}, m_meshes{meshes}, m_emitters{emitters} {
  // get all meshes and construct bounding mesh
  auto all_meshes = meshes;
  auto mtrls = std::vector<Material *>{};
  for (auto m : all_meshes) {
    mtrls.emplace_back(m->material().get());
  }
  auto lights = std::vector<Light *>(all_meshes.size(), nullptr);
  auto env_light = std::shared_ptr<Light>{nullptr};
  for (auto e : emitters) {
    if (e->mesh() != nullptr) {
      all_meshes.emplace_back(e->mesh());
      lights.emplace_back(e.get());
      mtrls.emplace_back(e->mesh()->material().get());
    }
    if (e->type() == LightType::envmap ||
        e->type() == LightType::spherical_harmonics) {
      if (env_light != nullptr) {
        std::cerr << "only support no more than one env_light";
        // TODO: throw exception
      }
      env_light = e;
    }
  }
  auto optim_meshes = all_meshes;
  if (env_light != nullptr) {
    auto tmp = cameras.at(0)->camera_pos<false>();
    Vec3fC pmin = enoki::detach(tmp);
    Vec3fC pmax = enoki::detach(tmp);
    for (auto c : cameras) {
      auto look_from_ = c->camera_pos<false>();
      auto look_from = enoki::detach(look_from_);
      pmin = enoki::min(pmin, look_from);
      pmax = enoki::max(pmax, look_from);
    }
    for (auto m : meshes) {
      auto [mesh_pmin, mesh_pmax] = m->aabb();
      pmin = enoki::min(pmin, mesh_pmin);
      pmax = enoki::max(pmax, mesh_pmax);
    }
    auto d = 1000 * enoki::hmax(pmax - pmin);
    auto c = (pmin + pmax) * Real{0.5};
    pmin = c - d;
    pmax = c + d;
    env_light->set_aabb(pmin, pmax);
    auto bounding_mesh = env_light->mesh();
    all_meshes.emplace_back(bounding_mesh);
    lights.emplace_back(env_light.get());
    mtrls.emplace_back(env_light->mesh()->material().get());
  }

  m_material = TypeD<Material *>::copy(mtrls.data(), mtrls.size());
  m_light = TypeD<Light *>::copy(lights.data(), lights.size());
  auto light_power = m_light->power();
  m_all_lights_power = enoki::hsum(light_power);

  auto triangle_offsets = std::vector<uint32_t>{0};
  for (auto mesh : all_meshes) {
    triangle_offsets.emplace_back(triangle_offsets.back() +
                                  mesh->triangles_cnt());
  }

  m_triangle = enoki::empty<TriangleD>(triangle_offsets.back());
  triangle_offsets.pop_back();
  for (auto i = 0u; i < all_meshes.size(); ++i) {
    auto mesh = all_meshes.at(i);
    auto offset = triangle_offsets.at(i);
    auto idx = enoki::arange<IntD>(mesh->triangles_cnt()) + offset;
    enoki::scatter(m_triangle, mesh->triangles(), idx);
  }

  m_light_idx_sampler =
      std::make_shared<AliasSampler>(enoki::detach(light_power));

  // construct object edges
  auto edges_cnt = size_t{0};
  for (auto mesh : optim_meshes) {
    edges_cnt += mesh->edges_cnt();
  }
  m_need_optimize_boundary = edges_cnt > 0;
  if (m_need_optimize_boundary) {
    m_object_edge = enoki::empty<EdgeD>(edges_cnt);
    auto offset = 0u;
    for (auto mesh : optim_meshes) {
      auto curr_cnt = mesh->edges_cnt();
      if (curr_cnt == 0) {
        continue;
      }
      auto idx = enoki::arange<IntD>(curr_cnt) + offset;
      enoki::scatter(m_object_edge, *mesh->edges(), idx);
      offset += curr_cnt;
    }
    m_edge_sampler =
        std::make_shared<AliasSampler>(enoki::detach(m_object_edge.length));
    m_all_edge_length = enoki::hsum(m_object_edge.length);
  }

  m_scene_optix = std::make_shared<SceneOptix>(all_meshes);
}

template <bool ad>
Intersection<ad> Scene::hit(const Ray<ad> &ray, const Mask<ad> &valid) const {
  auto optix_its = m_scene_optix->hit<ad>(ray, valid);
  auto &object_idx = optix_its.object_idx;
  auto &triangle_idx = optix_its.triangle_idx;
  auto &its_uv = optix_its.uv;
  auto new_valid = valid && object_idx >= 0 && triangle_idx >= 0;
  auto material = Type<Material *, ad>{};
  auto light = Type<Light *, ad>{};
  auto triangles = Triangle<ad>{};
  if constexpr (ad) {
    material =
        enoki::gather<TypeD<Material *>>(m_material, object_idx, new_valid);
    light = enoki::gather<TypeD<Light *>>(m_light, object_idx, new_valid);
    triangles = enoki::gather<TriangleD>(m_triangle, triangle_idx, new_valid);
  } else {
    material = enoki::gather<TypeC<Material *>>(enoki::detach(m_material),
                                                object_idx, new_valid);
    light = enoki::gather<TypeC<Light *>>(enoki::detach(m_light), object_idx,
                                          new_valid);
    triangles = enoki::gather<TriangleC>(enoki::detach(m_triangle),
                                         triangle_idx, new_valid);
  }

  auto p = triangles.p0 + its_uv.x() * (triangles.p1 - triangles.p0) +
           its_uv.y() * (triangles.p2 - triangles.p0);
  auto uv = triangles.uv0 + its_uv.x() * (triangles.uv1 - triangles.uv0) +
            its_uv.y() * (triangles.uv2 - triangles.uv0);

  auto wi_unnorm = p - ray.origin;
  auto wi = -enoki::normalize(wi_unnorm);
  auto squared_dist = enoki::squared_norm(wi_unnorm);
  auto is_front = new_valid && enoki::dot(wi, triangles.normal) > 0;

  auto normal = enoki::select(is_front, triangles.normal, -triangles.normal);
  auto tangent = enoki::select(is_front, triangles.tangent, -triangles.tangent);

  auto its = Intersection<ad>{};
  its.valid = std::move(new_valid);
  its.is_front = std::move(is_front);
  its.point = std::move(p);
  its.normal = std::move(normal);
  its.tangent = std::move(tangent);
  its.wi = std::move(wi);
  its.squared_dist = std::move(squared_dist);
  its.uv = std::move(uv);
  its.material = material;
  its.light = light;
  its.is_light = enoki::neq(light, nullptr);
  if constexpr (ad) {
    its.jacobi = triangles.area / enoki::detach(triangles.area);
  } else {
    its.jacobi = 1;
  }
  return its;
}

template <bool ad>
std::tuple<Type<Light *, ad>, Float<ad>>
Scene::sample_light_reuse(Float<ad> &sampler, const Mask<ad> &valid) const {
  auto [idx, pdf_light_choose] =
      m_light_idx_sampler->sample_reuse<ad>(sampler, valid);
  if constexpr (ad) {
    auto light = enoki::gather<TypeD<Light *>>(m_light, idx, valid);
    return {light, pdf_light_choose};
  } else {
    auto light =
        enoki::gather<TypeC<Light *>>(enoki::detach(m_light), idx, valid);
    return {light, pdf_light_choose};
  }
}

template <bool ad>
Float<ad> Scene::pdf_sampled_light(const Type<Light *, ad> &light,
                                   const Mask<ad> &valid) const {
  if constexpr (ad) {
    return enoki::select(valid, light->power() / m_all_lights_power, 0);
  } else {
    auto p = light->power() / enoki::detach(m_all_lights_power);
    return enoki::select(valid, enoki::detach(p), 0);
  }
}

template <bool ad>
std::tuple<Edge<ad>, Vec3f<ad>, Float<ad>>
Scene::sample_edge_point(const Float<ad> &sampler_,
                         const Mask<ad> &valid) const {
  auto sampler = sampler_;
  auto [idx, pdf] = m_edge_sampler->sample_reuse<ad>(sampler, valid);
  auto sampled_edge = Edge<ad>{};
  if constexpr (ad) {
    sampled_edge = enoki::gather<EdgeD>(m_object_edge, idx, valid);
  } else {
    sampled_edge =
        enoki::gather<EdgeC>(enoki::detach(m_object_edge), idx, valid);
  }
  auto sampled_point =
      sampled_edge.p0 + sampler * (sampled_edge.p1 - sampled_edge.p0);
  pdf /= sampled_edge.length;
  return {sampled_edge, sampled_point, pdf};
}

template IntersectionC Scene::hit<false>(const RayC &ray,
                                         const MaskC &valid) const;
template IntersectionD Scene::hit<true>(const RayD &ray,
                                        const MaskD &valid) const;

template std::tuple<TypeC<Light *>, FloatC>
Scene::sample_light_reuse<false>(FloatC &sampler, const MaskC &valid) const;
template std::tuple<TypeD<Light *>, FloatD>
Scene::sample_light_reuse<true>(FloatD &sampler, const MaskD &valid) const;

template FloatC Scene::pdf_sampled_light<false>(const TypeC<Light *> &light,
                                                const MaskC &valid) const;
template FloatD Scene::pdf_sampled_light<true>(const TypeD<Light *> &light,
                                               const MaskD &valid) const;

template std::tuple<EdgeC, Vec3fC, FloatC>
Scene::sample_edge_point<false>(const FloatC &sampler,
                                const MaskC &valid) const;
template std::tuple<EdgeD, Vec3fD, FloatD>
Scene::sample_edge_point<true>(const FloatD &sampler, const MaskD &valid) const;
