#pragma once

#include "optix.h"
#include "optix_stubs.h"
#include "src/cuda/params.h"
#include "src/mesh.h"
#include "src/optix_def.h"
#include "src/ray.h"
#include "src/types.h"

struct PathTracerState {
  OptixDeviceContext context = nullptr;
  OptixTraversableHandle gas_handle = 0;  // Traversable handle for triangle AS
  CUdeviceptr d_gas_output_buffer = 0;    // Triangle AS memory
  CUdeviceptr d_vertices = 0;
  OptixModule ptx_module = nullptr;
  OptixPipelineCompileOptions pipeline_compile_options = {};
  OptixPipeline pipeline = nullptr;
  OptixProgramGroup raygen_prog_group = nullptr;
  OptixProgramGroup radiance_miss_group = nullptr;
  OptixProgramGroup radiance_hit_group = nullptr;
  CUstream stream = 0;
  Params params;
  Params* d_params = nullptr;
  OptixShaderBindingTable sbt = {};

  ~PathTracerState() {
    release();
  }

 private:
  void release() {
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(radiance_miss_group));
    OPTIX_CHECK(optixProgramGroupDestroy(radiance_hit_group));
    OPTIX_CHECK(optixModuleDestroy(ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));
    // enoki::cuda_free(reinterpret_cast<void*>(d_gas_output_buffer));
  }
};

template <typename T>
struct Record {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

typedef Record<RayGenData> RayGenRecord;
typedef Record<MissData> MissRecord;
typedef Record<HitGroupData> HitGroupRecord;

template<bool ad>
struct IntersectionOptix {
  void reserve(size_t sz) {
    if(enoki::slices(uv) != sz) {
      triangle_idx = enoki::empty<Int<ad>>(sz);
      object_idx = enoki::empty<Int<ad>>(sz);
      uv = enoki::empty<Vec2f<ad>>(sz);
    }
  }
  Int<ad> object_idx;
  Int<ad> triangle_idx;
  Vec2f<ad> uv; // weight of edge1(p1-p0) ande edge2(p2-p0)
};

using IntersectionOptixC = IntersectionOptix<false>;
using IntersectionOptixD = IntersectionOptix<true>;

class SceneOptix {
 public:
  SceneOptix() = delete;
  explicit SceneOptix(const std::vector<std::shared_ptr<TriangleMesh>>& meshes);

  template <bool ad>
  IntersectionOptix<ad> hit(const Ray<ad>& ray,
                                      const Mask<ad>& valid) const;

 private:
  static void optix_config(PathTracerState& state,
                           const std::vector<uint32_t>& triangle_offsets);
  static void build_accel(PathTracerState& state,
                          const std::vector<OptixBuildInput>& triangle_input);

  std::shared_ptr<PathTracerState> m_accel;
};
