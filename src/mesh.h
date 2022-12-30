#pragma once

// #include "src/hittable.h"
#include "src/material.h"
#include "src/sampler.h"
#include "src/types.h"

template <typename Value> struct Triangle_ {
  Vec3<Value> p0;
  Vec3<Value> p1;
  Vec3<Value> p2;
  Vec2<Value> uv0;
  Vec2<Value> uv1;
  Vec2<Value> uv2;
  Vec3<Value> normal;
  Vec3<Value> tangent;
  Value area;
  ENOKI_STRUCT(Triangle_, p0, p1, p2, uv0, uv1, uv2, normal, tangent, area)
};
ENOKI_STRUCT_SUPPORT(Triangle_, p0, p1, p2, uv0, uv1, uv2, normal, tangent,
                     area)

template <bool ad> using Triangle = Triangle_<Float<ad>>;

using TriangleC = Triangle<false>;
using TriangleD = Triangle<true>;

template <typename Value> struct Edge_ {
  Vec3<Value> p0;
  Vec3<Value> p1;
  Vec3<Value> out_direction;
  Vec3<Value> n_left;
  Vec3<Value> n_right;
  Value length;
  ENOKI_STRUCT(Edge_, p0, p1, out_direction, n_left, n_right, length)
};
ENOKI_STRUCT_SUPPORT(Edge_, p0, p1, out_direction, n_left, n_right, length)

template <bool ad> using Edge = Edge_<Float<ad>>;

using EdgeC = Edge<false>;
using EdgeD = Edge<true>;

template <bool ad> struct SampledSurfacePoint {
  Float<ad> pdf;
  Vec3f<ad> point;
  Vec3f<ad> normal;  // always NdotV >= 0
  Vec3f<ad> tangent; // dP/du
  Vec2f<ad> uv;      // texture space coordinates
  Float<ad> jacobi;
  Mask<ad> is_env_light;
  Type<Material *, ad> material;
};

template <bool ad> struct SampledEdgePoint {
  Float<ad> pdf;
  Vec3f<ad> point;
  Edge<ad> edge;
};

using SampledSurfacePointC = SampledSurfacePoint<false>;
using SampledSurfacePointD = SampledSurfacePoint<true>;
using SampledEdgePointC = SampledEdgePoint<false>;
using SampledEdgePointD = SampledEdgePoint<true>;

class TriangleMesh final {
public:
  TriangleMesh() = delete;
  TriangleMesh(const Vec3fD &vertices, const Vec2fC &uvs,
               const Vec3iC &face_indices, const Vec3iC &uv_indices,
               std::shared_ptr<Material> material);

  size_t triangles_cnt() const { return m_triangles_cnt; }
  size_t vertices_cnt() const { return m_vertices_cnt; }
  size_t edges_cnt() const { return m_edges_cnt; }
  const FloatC &vertices_buffer() const { return m_vertices_buffer; }
  const IntC &faces_buffer() const { return m_faces_buffer; }
  const TriangleD &triangles() const { return m_triangles; }
  TriangleD &triangles() { return m_triangles; }
  std::shared_ptr<EdgeD> edges() const { return m_edges; }
  std::shared_ptr<Material> material() const { return m_material; }
  const FloatD &area() const { return m_total_area; }
  std::tuple<const Vec3fC &, const Vec3fC &> aabb() const {
    return {m_pmin, m_pmax};
  }

private:
  std::shared_ptr<EdgeD> build_edges(const Vec3iC &face_indices,
                                     const Vec3fD &vertices,
                                     const TriangleD &triangles);

  std::shared_ptr<Material> m_material;
  TriangleD m_triangles;
  std::shared_ptr<EdgeD> m_edges;
  FloatD m_total_area;
  Vec3fC m_pmin;
  Vec3fC m_pmax;
  std::shared_ptr<AliasSampler> m_face_position_sampler;
  size_t m_triangles_cnt;
  size_t m_vertices_cnt;
  size_t m_edges_cnt;
  bool m_has_uv;
  bool m_vertices_need_grad;

  // for optix ray tracing
  FloatC m_vertices_buffer;
  IntC m_faces_buffer;
};
