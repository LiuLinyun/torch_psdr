cc_library(
    name = "psdr",
    srcs = [
        "src/cuda/params.h",
        "src/types.h",
        "src/camera.h",
        "src/diffuse_bsdf.h",
        "src/diffuse_bsdf.cc",
        "src/diffuse_material.h",
        "src/diffuse_material.cc",
        "src/emitter.h",
        "src/emitter.cc",
        "src/integrator.h",
        "src/intersection.h",
        "src/material.h",
        "src/mesh.h",
        "src/mesh.cc",
        "src/optix_def.h",
        "src/path_integrator.h",
        "src/path_integrator.cc",
        "src/ray.h",
        "src/sampler.h",
        "src/scene.h",
        "src/scene.cc",
        "src/cuda/device_programs_data.h",
        "src/scene_optix.h",
        "src/scene_optix.cc",
        "src/texture.h",
    ],
    deps = [
        "@enoki//:enoki_cuda_autodiff",
        "@optix//:liboptix",
        "@cuda//:libcuda",
    ],
    linkstatic = False,
)

# load("@pybind11_bazel//:build_defs.bzl", "pybind_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name="pypsdr",
    srcs = [
        "src/psdr.cc",
    ],
    deps = [
        ":psdr",
    ],
    copts = [
        "-flto=thin",
        # "-fvisibility=hidden",
    ],
    linkopts = [
        "-flto=thin",
    ],
    linkstatic = False,
)

py_library(
    name = "torch_psdr",
    srcs = [
        "src/python/torch_psdr.py",
    ],
    data = [
        ":pypsdr.so",
    ],
    deps = [
        "@enoki//:enoki",
    ]
)

py_binary(
    name = "optim_torch",
    srcs = [
        "example/optim_torch.py",
    ],
    deps = [
        "torch_psdr",
    ],
)


py_binary(
    name = "opt_tex",
    srcs = [
        "example/opt_tex.py",
    ],
    deps = [
        ":torch_psdr",
    ],
)

py_binary(
    name = "optimize_points",
    srcs = [
        "example/optimize_points.py"
    ],
    deps = [
        ":torch_psdr",
    ],
)

cc_binary(
    name = "test_",
    srcs = [
        "example/test.cc",
    ],
    deps = [
        ":libdr",
    ],
)

cc_binary(
    name = "render",
    srcs = [
        "example/render.cc",
    ],
    deps = [
        ":libdr",
    ],
)

cc_binary(
    name = "optimize",
    srcs = [
        "example/optimize.cc",
    ],
    deps = [
        ":libdr",
    ],
)

cc_binary(
    name = "optim",
    srcs = [
        "example/optim.cc",
    ],
    deps = [
        ":libdr",
    ],
)

cc_binary(
    name = "test_enoki",
    srcs = [
        "example/test_enoki.cc",
    ],
    deps = [
        "@enoki//:enoki_cuda_autodiff",
    ],
)
