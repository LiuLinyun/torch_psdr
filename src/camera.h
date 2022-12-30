#pragma once

#include "src/ray.h"
#include "src/sampler.h"
#include "src/types.h"

template <bool ad>
Mat4f<ad> perspective(const Float<ad> &fov, const Float<ad> &near,
                      const Float<ad> &far) {
  auto inv_n_f = enoki::rcp(near - far);
  auto c = enoki::cot(fov * Real{0.5});
  auto mat = enoki::diag<Mat4f<ad>>(
      enoki::column_t<Mat4f<ad>>(-c, -c, (near + far) * inv_n_f, 0));
  mat(2, 3) = -2 * far * near * inv_n_f;
  mat(3, 2) = 1;
  return mat;
}

template <bool ad>
Mat4f<ad> orthographic(const Float<ad> &left, const Float<ad> &right,
                       const Float<ad> &bottom, const Float<ad> &top,
                       const Float<ad> &near, const Float<ad> &far) {
  auto trans = enoki::translate<Mat4f<ad>, Vec3f<ad>>(
      Vec3f<ad>{-(left + right) / 2, -(bottom + top) / 2, -(near + far) / 2});
  auto scale = enoki::scale<Mat4f<ad>, Vec3f<ad>>(Vec3f<ad>{
      2 / (right - left),
      2 / (top - bottom),
      2 / (near - far),
  });
  return scale * trans;
}

template <bool ad>
Mat4f<ad> look_at(const Vec3f<ad> &origin, const Vec3f<ad> &target,
                  const Vec3f<ad> &up) {
  auto dir = enoki::normalize(origin - target);
  auto right = enoki::cross(enoki::normalize(up), dir);
  auto new_up = enoki::cross(dir, right);
  using Scalar = enoki::scalar_t<Mat4f<ad>>;
  // auto T = enoki::translate<Mat4f<ad>, Vec3f<ad>>(-origin);
  // auto M = Mat4f<ad>::from_rows(
  //     enoki::concat(right, Scalar{0}),
  //     enoki::concat(new_up, Scalar{0}),
  //     enoki::concat(dir, Scalar{0}),
  //     enoki::concat(Vec3f<ad>{0}, Scalar{1}));
  // return M * T;
  return Mat4f<ad>{
      enoki::concat(right, Scalar{0}),
      enoki::concat(new_up, Scalar{0}),
      enoki::concat(dir, Scalar{0}),
      enoki::concat(origin, Scalar{1}),
  };
}

template <bool ad>
Vec3f<ad> transform_pos(const Mat4f<ad> &mat, const Vec3f<ad> &pos) {
  auto tmp = mat * enoki::concat(pos, Real{1});
  return enoki::head<3>(tmp) / tmp.w();
}

template <bool ad>
Vec3f<ad> transform_dir(const Mat4f<ad> &mat, const Vec3f<ad> &vec) {
  return enoki::head<3>(mat * enoki::concat(vec, Real{0}));
}

template <bool ad> class Film {
public:
  Film() = delete;
  Film(uint32_t height, uint32_t width)
      : m_height{height}, m_width{width}, m_pixel{enoki::zero<Spectrum<ad>>(
                                              height * width)} {}

  const Spectrum<ad> &data() const { return m_pixel; }

  void add(const Spectrum<ad> &data, const Vec2i<ad> &raster_coord,
           const Mask<ad> &valid = true, uint32_t spp = 1) {
    auto idx = raster_coord.y() * m_width + raster_coord.x();
    if (spp > 1) {
      auto n = enoki::slices(m_pixel);
      auto val = enoki::zero<Spectrum<ad>>(n);
      enoki::scatter_add(val, data, idx, valid);
      val /= Int<ad>{spp};
      m_pixel += val;
    } else {
      enoki::scatter_add(m_pixel, data, idx, valid);
    }
  }

  void div(int32_t d) {
    if (d == 1) {
      ;
    } else if (d == -1) {
      m_pixel *= -1;
    } else {
      m_pixel /= Int<ad>{d};
    }
  }

private:
  uint32_t m_height;
  uint32_t m_width;
  Spectrum<ad> m_pixel;
};

using FilmC = Film<false>;
using FilmD = Film<true>;

class BaseCamera {
public:
  BaseCamera() = delete;
  BaseCamera(const Mat4fD &camera2world, const Mat4fD &camera2screen,
             uint32_t height, uint32_t width)
      : m_height{height}, m_width{width}, m_camera2world{camera2world},
        m_camera2screen{camera2screen} {
    m_world2camera = enoki::inverse(m_camera2world);
    m_screen2raster = enoki::scale<Mat4fD, Vec3fD>(
                          Vec3fD{0.5 * m_width, -0.5 * m_height, 1}) *
                      enoki::translate<Mat4fD, Vec3fD>(Vec3fD{1, -1, 0});
    m_camera2raster = m_screen2raster * m_camera2screen;
    m_raster2camera = enoki::inverse(m_camera2raster);

    m_camera_position = transform_pos<true>(m_camera2world, Vec3fD{0, 0, 0});
    m_look_dir = transform_dir<true>(m_camera2world, Vec3fD{0, 0, -1});
  }
  virtual ~BaseCamera() = default;

  Sampler gen_sampler(uint32_t spp) const {
    auto n_samples = spp * m_height * m_width;
    return Sampler(n_samples);
  }

  template <bool ad> Vec3f<ad> camera_pos() const {
    if constexpr (ad) {
      return m_camera_position;
    } else {
      return enoki::detach(m_camera_position);
    }
  }

  template <bool ad> Vec3f<ad> look_dir() const {
    if constexpr (ad) {
      return m_look_dir;
    } else {
      return enoki::detach(m_look_dir);
    }
  }

  uint32_t height() const { return m_height; }
  uint32_t width() const { return m_width; }

  template <bool ad> Mat4f<ad> camera2world() const {
    if constexpr (ad) {
      return m_camera2world;
    } else {
      return enoki::detach(m_camera2world);
    }
  }

  template <bool ad> Mat4f<ad> world2camera() const {
    if constexpr (ad) {
      return m_world2camera;
    } else {
      return enoki::detach(m_world2camera);
    }
  }

  template <bool ad> Mat4f<ad> camera2raster() const {
    if constexpr (ad) {
      return m_camera2raster;
    } else {
      return enoki::detach(m_camera2raster);
    }
  }

  template <bool ad> Mat4f<ad> raster2camera() const {
    if constexpr (ad) {
      return m_raster2camera;
    } else {
      return enoki::detach(m_raster2camera);
    }
  }

  virtual std::tuple<RayC, Vec2iC>
  gen_camera_ray(const Vec2fC &sampler) const = 0;
  virtual std::tuple<RayD, Vec2iD>
  gen_camera_ray(const Vec2fD &sampler) const = 0;

  virtual std::tuple<RayC, Vec2iC, MaskC>
  gen_camera_ray_with_point(const Vec3fC &point,
                            const MaskC &valid = true) const = 0;
  virtual std::tuple<RayD, Vec2iD, MaskD>
  gen_camera_ray_with_point(const Vec3fD &point,
                            const MaskD &valid = true) const = 0;

protected:
  uint32_t m_height;
  uint32_t m_width;
  Vec3fD m_camera_position;
  Vec3fD m_look_dir;
  Mat4fD m_camera2world;
  Mat4fD m_world2camera;
  Mat4fD m_camera2screen;
  Mat4fD m_screen2raster;
  Mat4fD m_raster2camera;
  Mat4fD m_camera2raster;
};

class PerspectiveCamera final : public BaseCamera {
public:
  PerspectiveCamera(const Mat4fD &camera2world, const FloatD &fov, Real near,
                    Real far, uint32_t height, uint32_t width)
      : BaseCamera{camera2world, perspective<true>(fov, near, far), height,
                   width} {}

  std::tuple<RayC, Vec2iC>
  gen_camera_ray(const Vec2fC &sampler) const override {
    return gen_camera_ray_<false>(sampler);
  }
  std::tuple<RayD, Vec2iD>
  gen_camera_ray(const Vec2fD &sampler) const override {
    return gen_camera_ray_<true>(sampler);
  }
  std::tuple<RayC, Vec2iC, MaskC>
  gen_camera_ray_with_point(const Vec3fC &point,
                            const MaskC &valid = true) const override {
    return gen_camera_ray_with_point_<false>(point, valid);
  }
  std::tuple<RayD, Vec2iD, MaskD>
  gen_camera_ray_with_point(const Vec3fD &point,
                            const MaskD &valid = true) const override {
    return gen_camera_ray_with_point_<true>(point, valid);
  }

private:
  template <bool ad>
  std::tuple<Ray<ad>, Vec2i<ad>>
  gen_camera_ray_(const Vec2f<ad> &sampler) const {
    auto sampler_cnt = enoki::slices(sampler);
    auto spp = sampler_cnt / (m_width * m_height);
    auto idx = spp > 1 ? enoki::arange<Int<ad>>(sampler_cnt) / Int<ad>{spp}
                       : enoki::arange<Int<ad>>(sampler_cnt);
    auto raster_coord = enoki::gather<Vec2i<ad>>(
        enoki::meshgrid(enoki::arange<Int<ad>>(m_width),
                        enoki::arange<Int<ad>>(m_height)),
        idx);

    // get film coordinates
    auto film_xy = sampler + raster_coord;
    auto point = enoki::concat(film_xy, Real{1});
    // transform from raster space to camera space
    point = transform_pos<ad>(raster2camera<ad>(), point);
    // transform to world space
    // auto origin = transform_pos<ad>(camera2world, point);
    auto origin = camera_pos<ad>() + enoki::zero<Vec3f<ad>>(sampler_cnt);
    auto dir = enoki::normalize(transform_dir<ad>(camera2world<ad>(), point));
    auto ray = Ray<ad>{origin, dir};
    return {ray, raster_coord};
  }

  template <bool ad>
  std::tuple<Ray<ad>, Vec2i<ad>, Mask<ad>>
  gen_camera_ray_with_point_(const Vec3f<ad> &point,
                             const Mask<ad> &valid) const {
    // world to camera
    auto p_camera_space = transform_pos<ad>(world2camera<ad>(), point);
    // camera to raster
    auto p = transform_pos<ad>(camera2raster<ad>(), p_camera_space);
    auto new_valid = valid && p_camera_space.z() < 0 && p.x() > 0 &&
                     p.x() < m_width && p.y() > 0 && p.y() < m_height;

    auto raster_coord = enoki::floor2int<Vec2i<ad>>(enoki::head<2>(p));
    auto cam_pos =
        camera_pos<ad>() + enoki::zero<Vec3f<ad>>(enoki::slices(point));
    auto ray = Ray<ad>{cam_pos, enoki::normalize(point - cam_pos)};
    return {ray, raster_coord, new_valid};
  }
};

class OrthographicCamera final : public BaseCamera {
public:
  OrthographicCamera(const Mat4fD &camera2world, const FloatD& view_width,
                     uint32_t height, uint32_t width)
      : BaseCamera{camera2world, ortho_mat(view_width, height, width), height,
                   width} {}

  std::tuple<RayC, Vec2iC>
  gen_camera_ray(const Vec2fC &sampler) const override {
    return gen_camera_ray_<false>(sampler);
  }
  std::tuple<RayD, Vec2iD>
  gen_camera_ray(const Vec2fD &sampler) const override {
    return gen_camera_ray_<true>(sampler);
  }
  std::tuple<RayC, Vec2iC, MaskC>
  gen_camera_ray_with_point(const Vec3fC &point,
                            const MaskC &valid = true) const override {
    return gen_camera_ray_with_point_<false>(point, valid);
  }
  std::tuple<RayD, Vec2iD, MaskD>
  gen_camera_ray_with_point(const Vec3fD &point,
                            const MaskD &valid = true) const override {
    return gen_camera_ray_with_point_<true>(point, valid);
  }

private:
  Mat4fD ortho_mat(const FloatD& view_width, uint32_t height, uint32_t width) const {
    auto half_view_width = view_width / 2;
    auto half_view_height = height * half_view_width / width;
    auto left = -half_view_width;
    auto right = half_view_width;
    auto bottom = -half_view_height;
    auto top = half_view_height;
    auto near = Real{-0.01};
    auto far = Real{-1000};
    return orthographic<true>(left, right, bottom, top, near, far);
  }

  template <bool ad>
  std::tuple<Ray<ad>, Vec2i<ad>>
  gen_camera_ray_(const Vec2f<ad> &sampler) const {
    auto sampler_cnt = enoki::slices(sampler);
    auto spp = sampler_cnt / (m_width * m_height);
    auto idx = spp > 1 ? enoki::arange<Int<ad>>(sampler_cnt) / Int<ad>{spp}
                       : enoki::arange<Int<ad>>(sampler_cnt);
    auto raster_coord = enoki::gather<Vec2i<ad>>(
        enoki::meshgrid(enoki::arange<Int<ad>>(m_width),
                        enoki::arange<Int<ad>>(m_height)),
        idx);

    // get film coordinates
    auto film_xy = sampler + raster_coord;
    auto point = enoki::concat(film_xy, Real{1});
    // transform from raster space to camera space
    point = transform_pos<ad>(raster2camera<ad>(), point);
    // transform to world space
    auto origin = transform_pos<ad>(
        camera2world<ad>(), enoki::concat(enoki::head<2>(point), Real{0}));
    auto dir = look_dir<ad>() + enoki::zero<Vec3f<ad>>(sampler_cnt);
    auto ray = Ray<ad>{origin, dir};
    return {ray, raster_coord};
  }

  template <bool ad>
  std::tuple<Ray<ad>, Vec2i<ad>, Mask<ad>>
  gen_camera_ray_with_point_(const Vec3f<ad> &point,
                             const Mask<ad> &valid) const {
    // world to camera
    auto p_camera_space = transform_pos<ad>(world2camera<ad>(), point);
    // camera to raster
    auto p = transform_pos<ad>(camera2raster<ad>(), p_camera_space);
    auto new_valid = valid && p_camera_space.z() < 0 && p.x() > 0 &&
                     p.x() < m_width && p.y() > 0 && p.y() < m_height;

    auto raster_coord = enoki::floor2int<Vec2i<ad>>(enoki::head<2>(p));
    auto origin = transform_pos<ad>(
        camera2world<ad>(),
        enoki::concat(enoki::head<2>(p_camera_space), Real{0}));
    auto dir = look_dir<ad>() + enoki::zero<Vec3f<ad>>(enoki::slices(point));
    auto ray = Ray<ad>{origin, dir};
    return {ray, raster_coord, new_valid};
  }
};
