#include "enoki/python.h"
#include "pybind11/complex.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "src/camera.h"
#include "src/diffuse_bsdf.h"
#include "src/diffuse_material.h"
#include "src/mesh.h"
#include "src/path_integrator.h"
#include "src/types.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(pypsdr, m) {
  py::module::import("enoki");
  py::module::import("enoki.cuda");
  py::module::import("enoki.cuda_autodiff");

  py::class_<BaseCamera, std::shared_ptr<BaseCamera>>(m, "BaseCamera");

  py::class_<PerspectiveCamera, BaseCamera, std::shared_ptr<PerspectiveCamera>>(
      m, "PerspectiveCamera")
      .def(py::init([](Vec3fD look_from, Vec3fD look_at_, Vec3fD up,
                       FloatD vfov, uint32_t height, uint32_t width) {
        auto camera2world = look_at<true>(look_from, look_at_, up);
        return PerspectiveCamera{camera2world, vfov,   Real{-0.01},
                                 Real{-1000},  height, width};
      }));

  py::class_<OrthographicCamera, BaseCamera,
             std::shared_ptr<OrthographicCamera>>(m, "OrthographicCamera")
      .def(py::init([](Vec3fD look_from, Vec3fD look_at_, Vec3fD up,
                       FloatD view_width, uint32_t height, uint32_t width) {
        auto camera2world = look_at<true>(look_from, look_at_, up);
        return OrthographicCamera{camera2world, view_width, height, width};
      }));

  py::class_<Texture1f, std::shared_ptr<Texture1f>>(m, "Texture1f");

  py::class_<Texture3f, std::shared_ptr<Texture3f>>(m, "Texture3f");

  py::class_<ConstantTexture1f, Texture1f, std::shared_ptr<ConstantTexture1f>>(
      m, "ConstantTexture1f")
      .def(py::init<const FloatD &>(), "val"_a);

  py::class_<ConstantTexture3f, Texture3f, std::shared_ptr<ConstantTexture3f>>(
      m, "ConstantTexture3f")
      .def(py::init<const Vec3fD &>(), "val"_a);

  py::class_<UVTexture1f, Texture1f, std::shared_ptr<UVTexture1f>>(
      m, "UVTexture1f")
      .def(py::init<uint32_t, uint32_t, const FloatD &>(), "height"_a,
           "width"_a, "data"_a);

  py::class_<UVTexture3f, Texture3f, std::shared_ptr<UVTexture3f>>(
      m, "UVTexture3f")
      .def(py::init<uint32_t, uint32_t, const Vec3fD &>(), "height"_a,
           "width"_a, "data"_a);

  py::class_<Material, std::shared_ptr<Material>>(m, "Material");

  py::class_<DiffuseMaterial, Material, std::shared_ptr<DiffuseMaterial>>(
      m, "DiffuseMaterial")
      .def(py::init<const Vec3fD &>(), "tex"_a)
      .def(py::init<std::shared_ptr<Texture3f>>(), "tex"_a);

  py::class_<DiffuseLightMaterial, DiffuseMaterial,
             std::shared_ptr<DiffuseLightMaterial>>(m, "DiffuseLightMaterial")
      .def(py::init<const Vec3fD &>(), "tex"_a)
      .def(py::init<std::shared_ptr<Texture3f>>(), "tex"_a);

  py::class_<DiffuseBsdfMaterial, Material,
             std::shared_ptr<DiffuseBsdfMaterial>>(m, "DiffuseBsdfMaterial")
      .def(py::init<std::shared_ptr<Texture3f>, std::shared_ptr<Texture1f>,
                    std::shared_ptr<Texture3f>>(),
           "color"_a, "roughness"_a, "normal"_a.none());

  py::class_<TriangleMesh, std::shared_ptr<TriangleMesh>>(m, "TriangleMesh")
      .def(py::init([](const Vec3fD &vertices, const Vec2fC &uvs,
                       const Vec3iC &face_indices, const Vec3iC &uv_indices,
                       std::shared_ptr<Material> material) {
        return TriangleMesh{vertices, uvs, face_indices, uv_indices, material};
      }));

  py::class_<Light, std::shared_ptr<Light>>(m, "Light");

  py::class_<AreaLight, Light, std::shared_ptr<AreaLight>>(m, "AreaLight")
      .def(py::init([](const Vec3fD &vertices, const Vec3iC &face_indices,
                       const Vec3fD &emit) {
        return AreaLight{vertices, face_indices, emit};
      }));

  py::class_<EnvLight, Light, std::shared_ptr<EnvLight>>(m, "EnvLight")
      .def(py::init<std::shared_ptr<UVTexture3f>>(), "cube_map_tex"_a);

  py::class_<SH9Light, Light, std::shared_ptr<SH9Light>>(m, "SH9Light")
      .def(py::init<const Vec3fD &>(), "sh9_coeff"_a);

  py::class_<Scene>(m, "Scene")
      .def(
          py::init([](const std::vector<std::shared_ptr<BaseCamera>> &cameras,
                      const std::vector<std::shared_ptr<TriangleMesh>> &objects,
                      const std::vector<std::shared_ptr<Light>> &lights) {
            return Scene{cameras, objects, lights};
          }));

  py::class_<PathIntegrator::Config>(m, "PathIntegratorConfig")
      .def(py::init<>())
      .def_readwrite("n_pass", &PathIntegrator::Config::n_pass)
      .def_readwrite("spp_interior", &PathIntegrator::Config::spp_interior)
      .def_readwrite("enable_light_visable",
                     &PathIntegrator::Config::enable_light_visable)
      .def_readwrite("spp_primary_edge",
                     &PathIntegrator::Config::spp_primary_edge)
      .def_readwrite("spp_secondary_edge",
                     &PathIntegrator::Config::spp_secondary_edge)
      .def_readwrite("max_bounce", &PathIntegrator::Config::max_bounce)
      .def_readwrite("mis_light_samples",
                     &PathIntegrator::Config::mis_light_samples)
      .def_readwrite("mis_bsdf_samples",
                     &PathIntegrator::Config::mis_bsdf_samples)
      .def_readwrite("primary_edge_preprocess_rounds",
                     &PathIntegrator::Config::primary_edge_preprocess_rounds)
      .def_readwrite("secondary_edge_preprocess_rounds",
                     &PathIntegrator::Config::secondary_edge_preprocess_rounds)
      .def_readwrite("primary_edge_hypercube_resolution",
                     &PathIntegrator::Config::primary_edge_hypercube_resolution)
      .def_readwrite(
          "secondary_edge_hypercube_resolution",
          &PathIntegrator::Config::secondary_edge_hypercube_resolution);

  py::class_<PathIntegrator>(m, "PathIntegrator")
      .def(py::init<>())
      .def(py::init<const PathIntegrator::Config &>(), "config"_a)
      .def("renderC", &PathIntegrator::renderC)
      .def("renderD", &PathIntegrator::renderD);
}