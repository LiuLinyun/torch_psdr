#include "src/mesh.h"

#include <unordered_map>

TriangleMesh::TriangleMesh(const Vec3fD &vertices, const Vec2fC &uvs_,
                           const Vec3iC &face_indices, const Vec3iC &uv_indices,
                           std::shared_ptr<Material> material)
    : m_material{material} {
  // construct triangles
  m_triangles_cnt = enoki::slices(face_indices);
  m_vertices_cnt = enoki::slices(vertices);
  m_has_uv = enoki::slices(uvs_) != 0;
  // NOTE: to avoid optimize camera position faild without grad, always turn on vertices grad switch
  // TODO: find a method to avoid boundary render is unnecessary
  // m_vertices_need_grad = enoki::requires_gradient(vertices);
  m_vertices_need_grad = true;

  auto vertices_ = enoki::detach(vertices);
  m_pmin = Vec3fC{enoki::hmin(vertices_.x()), enoki::hmin(vertices_.y()),
                  enoki::hmin(vertices_.z())};
  m_pmax = Vec3fC{enoki::hmax(vertices_.x()), enoki::hmax(vertices_.y()),
                  enoki::hmax(vertices_.z())};

  auto face_idx = Vec3iD{face_indices};
  auto uv_idx = Vec3iD{uv_indices};

  m_triangles.p0 = enoki::gather<Vec3fD>(vertices, face_idx.x());
  m_triangles.p1 = enoki::gather<Vec3fD>(vertices, face_idx.y());
  m_triangles.p2 = enoki::gather<Vec3fD>(vertices, face_idx.z());
  auto uvs = Vec2fD{uvs_};
  m_triangles.uv0 =
      m_has_uv ? enoki::gather<Vec2fD>(uvs, uv_idx.x()) : Vec2fD{0, 0};
  m_triangles.uv1 =
      m_has_uv ? enoki::gather<Vec2fD>(uvs, uv_idx.y()) : Vec2fD{0, 1};
  m_triangles.uv2 =
      m_has_uv ? enoki::gather<Vec2fD>(uvs, uv_idx.z()) : Vec2fD{1, 0};

  auto a = m_triangles.p2 - m_triangles.p1;
  auto b = m_triangles.p0 - m_triangles.p2;
  auto c = m_triangles.p1 - m_triangles.p0;

  m_triangles.normal = enoki::normalize(enoki::cross(c, b));

  {
    auto duv1 = m_triangles.uv1 - m_triangles.uv0;
    auto duv2 = m_triangles.uv2 - m_triangles.uv0;
    // auto mask = enoki::norm(duv1) < 1e5*REAL_EPS ||
    //             enoki::norm(duv2) < 1e5*REAL_EPS;
    // duv1[mask] = Vec2fD{0, 1};
    // duv2[mask] = Vec2fD{1, 0};
    // auto duv1 = Vec2fD{0,1};
    // auto duv2 = Vec2fD{1,0};
    const auto &e1 = c;
    const auto &e2 = b;
    // auto ratio = enoki::rcp(vec2_cross<true>(duv1, duv2));
    m_triangles.tangent =
        enoki::normalize(duv2.y() * e1 + duv1.y() * e2);
  }

  {
    auto a_len = enoki::norm(a);
    auto b_len = enoki::norm(b);
    auto c_len = enoki::norm(c);
    auto p = (a_len + b_len + c_len) / 2;
    m_triangles.area =
        enoki::safe_sqrt(p * (p - a_len) * (p - b_len) * (p - c_len));
  }

  // construct edges
  if (m_vertices_need_grad) {
    m_edges = build_edges(face_indices, vertices, m_triangles);
    m_edges_cnt = enoki::slices(*m_edges);
  } else {
    m_edges_cnt = 0;
  }

  m_total_area = enoki::hsum(m_triangles.area);
  m_face_position_sampler =
      std::make_shared<AliasSampler>(enoki::detach(m_triangles.area));

  // construct optix buffer, ordered one by one
  {
    m_vertices_buffer = enoki::empty<FloatC>(m_vertices_cnt * 3);
    auto v_idx = enoki::arange<IntC>(m_vertices_cnt) * 3;
    for (auto i = 0; i < 3; ++i) {
      enoki::scatter(m_vertices_buffer, enoki::detach(vertices[i]), v_idx + i);
    }

    m_faces_buffer = enoki::empty<IntC>(m_triangles_cnt * 3);
    auto f_idx = enoki::arange<IntC>(m_triangles_cnt) * 3;
    for (auto i = 0; i < 3; ++i) {
      enoki::scatter(m_faces_buffer, face_indices[i], f_idx + i);
    }
  }
}

std::shared_ptr<EdgeD> TriangleMesh::build_edges(const Vec3iC &face_indices,
                                                 const Vec3fD &vertices,
                                                 const TriangleD &triangles) {
  struct EdgeIdx {
    EdgeIdx(Integer i_, Integer j_) : i{i_}, j{j_} {
      if (j < i) {
        std::swap(i, j);
      }
    }
    bool operator==(const EdgeIdx &other) const {
      return i == other.i && j == other.j;
    }

    Integer i;
    Integer j;
  };

  struct FaceIdx {
    FaceIdx(Integer l_, Integer r_, Integer kl_, Integer kr_)
        : l{l_}, r{r_}, kl{kl_}, kr{kr_} {}
    Integer l;
    Integer r;
    Integer kl; // third point of left face
    Integer kr; // third point or right face
  };

  auto hash_fn = [](const EdgeIdx &e) {
    auto hasher = std::hash<Integer>{};
    return hasher(e.i) ^ hasher(e.j);
  };

  auto face_cnt = enoki::slices(face_indices);
  auto edge_idx_info = std::unordered_map<EdgeIdx, FaceIdx, decltype(hash_fn)>{
      face_cnt * 3, hash_fn};

  auto add_edge_fn = [&edge_idx_info](int32_t i, int32_t j, int32_t k,
                                      int32_t face_idx) {
    auto key = EdgeIdx{i, j};

    if (edge_idx_info.contains(key)) {
      auto &val = edge_idx_info.at(key);
      val.r = face_idx;
      val.kr = k;
    } else {
      edge_idx_info.emplace(key, FaceIdx{face_idx, -1, k, -1});
    }
  };

  for (auto face_idx = 0; face_idx < face_cnt; ++face_idx) {
    auto i = face_indices[0][face_idx];
    auto j = face_indices[1][face_idx];
    auto k = face_indices[2][face_idx];
    add_edge_fn(i, j, k, face_idx);
    add_edge_fn(j, k, i, face_idx);
    add_edge_fn(k, i, j, face_idx);
  }

  auto tmp_data = std::array<std::vector<Integer>, 6>{};
  for (const auto &[k, v] : edge_idx_info) {
    tmp_data[0].emplace_back(k.i);                   // p0_idx;
    tmp_data[1].emplace_back(k.j);                   // p1_idx;
    tmp_data[2].emplace_back(v.l);                   // left face idx
    tmp_data[3].emplace_back(v.r == -1 ? v.l : v.r); // right face idx
    tmp_data[4].emplace_back(v.kl); // third point idx of left face
    tmp_data[5].emplace_back(v.r == -1 ? v.kl
                                       : v.kr); // right point idx of right face
  }

  auto edges_cnt = tmp_data[0].size();
  auto p0_idx = IntD{IntC::copy(tmp_data[0].data(), edges_cnt)};
  auto p1_idx = IntD{IntC::copy(tmp_data[1].data(), edges_cnt)};
  auto p2_left_idx = IntD{IntC::copy(tmp_data[4].data(), edges_cnt)};
  auto p2_right_idx = IntD{IntC::copy(tmp_data[5].data(), edges_cnt)};
  auto left_triangle_idx = IntD{IntC::copy(tmp_data[2].data(), edges_cnt)};
  auto right_triangle_idx = IntD{IntC::copy(tmp_data[3].data(), edges_cnt)};
  auto mask_unshared = enoki::neq(left_triangle_idx, right_triangle_idx);

  auto edge = std::make_shared<EdgeD>();
  edge->p0 = enoki::gather<Vec3fD>(vertices, p0_idx);
  edge->p1 = enoki::gather<Vec3fD>(vertices, p1_idx);

  auto p2_left = enoki::gather<Vec3fD>(vertices, p2_left_idx);
  auto p2_right = enoki::gather<Vec3fD>(vertices, p2_right_idx);
  auto n_left =
      enoki::normalize(enoki::cross(edge->p1 - edge->p0, p2_left - edge->p1));
  auto n_right =
      enoki::normalize(enoki::cross(p2_right - edge->p1, edge->p1 - edge->p0));
  auto out_dir = enoki::select(
      mask_unshared, enoki::normalize(n_left + n_right),
      enoki::normalize(enoki::cross(edge->p1 - edge->p0, n_left)));

  {
    auto mid_p = (edge->p0 + edge->p1);
    auto mid_c = (p2_left + p2_right);
    auto cp = enoki::normalize(mid_p - mid_c);
    auto need_reverse = enoki::dot(cp, out_dir) < 0 && mask_unshared &&
                        enoki::norm(n_left - n_right) > REAL_EPS;
    n_left[need_reverse] *= -1;
    n_right[need_reverse] *= -1;
    out_dir[need_reverse] *= -1;
  }

  edge->n_left = std::move(n_left);
  edge->n_right = std::move(n_right);
  edge->out_direction = std::move(out_dir);
  edge->length = enoki::norm(edge->p0 - edge->p1);
  return edge;
}
