cc_library(
  name = "enoki_hdrs",
  hdrs = glob([
    "include/enoki/*.h",
  ]),
  includes = [
    "include",
  ],
  copts = [
    # "-fvisibility=hidden",
  ],
  linkstatic = False,
)

cc_library(
  name = "enoki_cuda",
  hdrs = [
    "src/cuda/common.cuh",
  ],
  srcs = [
    "src/cuda/common.cu",
    "src/cuda/jit.cu",
    "src/cuda/horiz.cu",
  ],
  deps = [
    ":enoki_hdrs",
    "@cuda//:libcuda",
  ],
  copts = [
    "--cuda-gpu-arch=sm_61",
    "-fno-math-errno",
    "-fno-stack-protector",
    "-fomit-frame-pointer",
    "-std=c++17",
    # "-fvisibility=hidden",
  ],
  linkopts = [
    "-ldl",
    "-lrt",
    "-pthread",
  ],
  local_defines = [
    "ENOKI_CUDA",
    "ENOKI_AUTODIFF",
    "ENOKI_CUDA_COMPUTE_CAPABILITY=61",
    "ENOKI_AUTODIFF_BUILD",
    "ENOKI_BUILD",
    "NDEBUG",
  ],
  linkstatic = False,
)

cc_library(
  name = "enoki_cuda_autodiff",
  srcs = [
    "src/autodiff/autodiff.cpp",
  ],
  deps = [
    ":enoki_cuda",
  ],
  copts = [
    # "-fvisibility=hidden",
  ],
  linkopts = [
  ],
  local_defines = [
    "ENOKI_CUDA",
    "ENOKI_AUTODIFF",
    "THRUST_IGNORE_CUB_VERSION_CHECK",
    "ENOKI_AUTODIFF_BUILD",
    "ENOKI_BUILD",
    "NDEBUG",
  ],
  linkstatic = False,
  visibility = ["//visibility:public"],
)


load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
pybind_extension(
  name = "core",
  srcs = [
    "src/python/common.h",
    "src/python/main.cpp",
  ],
  deps = [
    ":enoki_hdrs",
  ],
  copts = [
    "-flto=thin",
    # "-fvisibility=hidden",
  ],
  linkopts = [
    "-flto=thin",
  ],
  local_defines = [
    "ENOKI_CUDA",
    "ENOKI_AUTODIFF",
  ],
  linkstatic = False,
)

pybind_extension(
  name = "scalar",
  srcs = [
    "src/python/common.h",
    "src/python/random.h",
    "src/python/complex.h",
    "src/python/matrix.h",
    "src/python/docstr.h",
    "src/python/quat.h",
    "src/python/scalar.cpp",
    "src/python/scalar_0d.cpp",
    "src/python/scalar_1d.cpp",
    "src/python/scalar_2d.cpp",
    "src/python/scalar_3d.cpp",
    "src/python/scalar_4d.cpp",
    "src/python/scalar_complex.cpp",
    "src/python/scalar_matrix.cpp",
    "src/python/scalar_quat.cpp",
    "src/python/scalar_pcg32.cpp",
  ],
  includes = [
    "src/python",
  ],
  deps = [
    ":enoki_hdrs",
  ],
  copts = [
    "-flto=thin",
    # "-fvisibility=hidden",
  ],
  linkopts = [
    "-flto=thin",
  ],
  local_defines = [
    "ENOKI_CUDA",
    "ENOKI_AUTODIFF",
  ],
  linkstatic = False,
)

pybind_extension(
  name = "dynamic",
  srcs = [
    "src/python/common.h",
    "src/python/random.h",
    "src/python/docstr.h",
    "src/python/complex.h",
    "src/python/matrix.h",
    "src/python/quat.h",
    "src/python/dynamic.cpp",
    "src/python/dynamic_0d.cpp",
    "src/python/dynamic_1d.cpp",
    "src/python/dynamic_2d.cpp",
    "src/python/dynamic_3d.cpp",
    "src/python/dynamic_4d.cpp",
    "src/python/dynamic_complex.cpp",
    "src/python/dynamic_matrix.cpp",
    "src/python/dynamic_pcg32.cpp",
  ],
  deps = [
    ":enoki_hdrs",
  ],
  copts = [
    "-flto=thin",
    # "-fvisibility=hidden",
  ],
  linkopts = [
    "-flto=thin",
  ],
  local_defines = [
    "ENOKI_CUDA",
    "ENOKI_AUTODIFF",
  ],
  linkstatic = False,
)

pybind_extension(
  name = "cuda",
  srcs = [
    "src/python/common.h",
    "src/python/random.h",
    "src/python/docstr.h",
    "src/python/complex.h",
    "src/python/matrix.h",
    "src/python/quat.h",
    "src/python/cuda.cpp",
    "src/python/cuda_0d.cpp",
    "src/python/cuda_1d.cpp",
    "src/python/cuda_2d.cpp",
    "src/python/cuda_3d.cpp",
    "src/python/cuda_4d.cpp",
    "src/python/cuda_complex.cpp",
    "src/python/cuda_matrix.cpp",
    "src/python/cuda_pcg32.cpp",
  ],
  deps = [
    ":enoki_cuda_autodiff",
  ],
  copts = [
    "-flto=thin",
    # "-fvisibility=hidden",
  ],
  linkopts = [
    "-flto=thin",
  ],
  local_defines = [
    "ENOKI_CUDA",
    "ENOKI_AUTODIFF",
  ],
  linkstatic = False,
)

pybind_extension(
  name = "cuda_autodiff",
  srcs = [
    "src/python/common.h",
    "src/python/complex.h",
    "src/python/matrix.h",
    "src/python/quat.h",
    "src/python/cuda_autodiff.cpp",
    "src/python/cuda_autodiff_0d.cpp",
    "src/python/cuda_autodiff_1d.cpp",
    "src/python/cuda_autodiff_2d.cpp",
    "src/python/cuda_autodiff_3d.cpp",
    "src/python/cuda_autodiff_4d.cpp",
    "src/python/cuda_autodiff_complex.cpp",
    "src/python/cuda_autodiff_matrix.cpp",
  ],
  deps = [
    ":enoki_cuda_autodiff",
  ],
  copts = [
    "-flto=thin",
    # "-fvisibility=hidden",
  ],
  linkopts = [
    "-flto=thin",
  ],
  local_defines = [
    "ENOKI_CUDA",
    "ENOKI_AUTODIFF",
  ],
  linkstatic = False,
)

py_library(
  name = "enoki",
  srcs = [],
  data = [
    ":core.so",
    ":scalar.so",
    ":dynamic.so",
    ":cuda.so",
    ":cuda_autodiff.so",
  ],
  visibility = ["//visibility:public"],
)