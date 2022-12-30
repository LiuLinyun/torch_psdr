cc_library(
  name = "libcuda",
  hdrs = glob([
    "include/**/*.h*",
    "include/**/*.cuh",
    "include/**/*.inl",
  ]),
  srcs = glob([
    "lib64/*.so*",
    "lib64/*.a",
  ]),
  includes = [
    "include",
  ],
  visibility = ["//visibility:public"],
)