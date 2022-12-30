
## Introduction

TorchPSDR is a physically based renderer.

## Installation

### Requirements

* Linux (Only Linux is supported now)
* python 3.8, 3.9 or 3.10
* pytorch (cuda based)

To run example, it's also need to install following libraries:
* imageio
* pywavefront_uv (conda install -c luling pywavefront_uv)

### Installing prebuilt binaries with conda

```bash
conda install -c luling torch_psdr
```

## Compiling

Before compiling this project, it's required to install:
* bazel >= 5.0
* clang >= 14.0
* nvidia cuda
* Optix

Then, set environment viriable `CUDA_PATH` and `OPTIX_PATH` in .bazelrc or specify them in build shell.

Finally, run
```bash
bazel build torch_psdr
```

**Please keep Internet connection during building this project, because it will fetch dependicies from internet**

## License
MIT

## Tutorials

1. [Hello Torch PSDR](example/render_scene.ipynb)
2. [Optimize Texture](example/optimize_texture.ipynb)
3. [Optimize Object Location](example/optimize_location.ipynb)
4. [Optimize Mesh Vertices and Texture](example/optimize_vertices.ipynb)

## API Reference
