cc_library(
  name = "liboptix",
  hdrs = glob([
    "include/**/*.h*",
  ]),
  includes = [
    "include",
  ],
  deps = [
    "@cuda//:libcuda",
  ],
  visibility = ["//visibility:public"],
)