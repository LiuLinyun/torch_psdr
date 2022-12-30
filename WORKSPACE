load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",

    # Replace the commit hash in both places (below) with the latest, rather than using the stale one here.
    # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/refs/heads/main.zip",
    strip_prefix = "bazel-compile-commands-extractor-main",
    # When you first run this tool, it'll recommend a sha256 hash to put here with a message like: "DEBUG: Rule 'hedron_compile_commands' indicated that a canonical reproducible form can be obtained by modifying arguments sha256 = ..."
)
load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()

http_archive(
  name = "enoki",
  build_file = "//:bazel_deps/enoki.BUILD",
  strip_prefix = "enoki-master",
  urls = ["https://github.com/mitsuba-renderer/enoki/archive/refs/heads/master.zip"],
)

load("//:bazel_deps/defs.bzl", "new_local_repository_env")

new_local_repository_env(
    name = "optix",
    build_file = "//:bazel_deps/optix.BUILD",
    path_var_name = "OPTIX_PATH",
)

new_local_repository_env(
    name = "cuda",
    build_file = "//:bazel_deps/cuda.BUILD",
    path_var_name = "CUDA_PATH",
)

# load pybind11_bazel
http_archive(
  name = "pybind11_bazel",
  strip_prefix = "pybind11_bazel-master",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/master.zip"],
)


# We still require the pybind library.
http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.10.1",
  urls = ["https://github.com/pybind/pybind11/archive/v2.10.1.tar.gz"],
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")
