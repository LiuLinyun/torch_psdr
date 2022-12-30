#include "src/scene_optix.h"
#include "src/cuda/device_programs_data.h"

#include <fstream>
#include <map>
#include <sstream>

#include "optix_function_table_definition.h"

SceneOptix::SceneOptix(
    const std::vector<std::shared_ptr<TriangleMesh>> &meshes) {
  auto num_meshes = meshes.size();
  auto triangle_offsets = std::vector<uint32_t>(num_meshes);
  triangle_offsets.at(0) = 0;
  for (auto i = 1u; i < num_meshes; ++i) {
    triangle_offsets.at(i) =
        triangle_offsets.at(i - 1) + meshes.at(i - 1)->triangles_cnt();
  }

  m_accel = std::make_shared<PathTracerState>();
  optix_config(*m_accel, triangle_offsets);

  auto triangle_input_flags =
      std::vector<unsigned int>{OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
  auto vertex_buffer_ptrs = std::vector<CUdeviceptr>(num_meshes);
  auto build_inputs = std::vector<OptixBuildInput>(num_meshes);
  for (auto i = 0u; i < num_meshes; ++i) {
    auto mesh = meshes.at(i);
    build_inputs.at(i).type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    auto &tri_arr = build_inputs.at(i).triangleArray;
    tri_arr.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    tri_arr.vertexStrideInBytes = 3 * sizeof(Real);
    tri_arr.numVertices = mesh->vertices_cnt();
    vertex_buffer_ptrs.at(i) =
        reinterpret_cast<CUdeviceptr>(mesh->vertices_buffer().data());
    tri_arr.vertexBuffers = &vertex_buffer_ptrs.at(i);
    tri_arr.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    tri_arr.numIndexTriplets = mesh->triangles_cnt();
    tri_arr.indexBuffer =
        reinterpret_cast<CUdeviceptr>(mesh->faces_buffer().data());
    tri_arr.indexStrideInBytes = 3 * sizeof(Integer);
    tri_arr.flags = triangle_input_flags.data();
    tri_arr.numSbtRecords = triangle_input_flags.size();
  }
  build_accel(*m_accel, build_inputs);
}

void SceneOptix::optix_config(PathTracerState &state,
                              const std::vector<uint32_t> &triangle_offsets) {
  enoki::cuda_eval();
  // step 0: init optix
  CUDA_CHECK(cudaFree(0));
  auto num_devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  if (num_devices == 0) {
    std::cerr << "no cuda device!" << std::endl;
    return;
  }
  OPTIX_CHECK(optixInit());

  // step 1: cerate context
  auto options = OptixDeviceContextOptions{};
  auto cu_ctx = CUcontext{nullptr};
  OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &state.context));

  // step 2: create module
  auto module_compile_options = OptixModuleCompileOptions{};
  module_compile_options.maxRegisterCount =
      OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  state.pipeline_compile_options.usesMotionBlur = false;
  state.pipeline_compile_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  state.pipeline_compile_options.numPayloadValues = 2;
  state.pipeline_compile_options.numAttributeValues = 2;
  state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

  auto ptx_str = reinterpret_cast<char *>(src_cuda_device_programs_ptx);
  auto ptx_str_size = src_cuda_device_programs_ptx_len;
  auto log = std::string(2048, ' ');
  auto sizeof_log = log.size();
  OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
      state.context,                   // context
      &module_compile_options,         // moduleCompileOptions
      &state.pipeline_compile_options, // pipelineCompileOptions
      ptx_str, ptx_str_size,           // ptx
      &log[0], &sizeof_log,            // log
      &state.ptx_module                // module
      ));

  // step 3: create program groups
  auto program_group_options = OptixProgramGroupOptions{};
  {
    sizeof_log = log.size();
    auto raygen_desc = OptixProgramGroupDesc{};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = state.ptx_module;
    raygen_desc.raygen.entryFunctionName = "__raygen__fn";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context, &raygen_desc, 1, &program_group_options, &log[0],
        &sizeof_log, &state.raygen_prog_group));
  }

  {
    sizeof_log = log.size();
    auto miss_desc = OptixProgramGroupDesc{};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = state.ptx_module;
    miss_desc.miss.entryFunctionName = "__miss__fn";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context, &miss_desc, 1, &program_group_options, &log[0],
        &sizeof_log, &state.radiance_miss_group));
  }

  {
    auto hit_desc = OptixProgramGroupDesc{};
    hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_desc.hitgroup.moduleCH = state.ptx_module;
    hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__fn";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context, &hit_desc, 1, &program_group_options, &log[0],
        &sizeof_log, &state.radiance_hit_group));
  }

  // step 4: create pipeline
  auto program_groups = std::vector<OptixProgramGroup>{
      state.raygen_prog_group, state.radiance_miss_group,
      state.radiance_hit_group};
  auto pipeline_link_options = OptixPipelineLinkOptions{};
  pipeline_link_options.maxTraceDepth = 1;
  pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  sizeof_log = log.size();
  OPTIX_CHECK_LOG(optixPipelineCreate(
      state.context, &state.pipeline_compile_options, &pipeline_link_options,
      program_groups.data(), program_groups.size(), &log[0], &sizeof_log,
      &state.pipeline));

  // step 5: create shader binding table(SBT)
  auto cuda_dptr = CUdeviceptr{};
  auto total_mem_size = sizeof(RayGenRecord) + sizeof(MissRecord) +
                        triangle_offsets.size() * sizeof(HitGroupRecord) +
                        sizeof(Params);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&cuda_dptr), total_mem_size));

  auto raygen_dptr = cuda_dptr;
  auto raygen_sbt = RayGenRecord{};
  OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &raygen_sbt));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(raygen_dptr), &raygen_sbt,
                        sizeof(RayGenRecord), cudaMemcpyHostToDevice));

  auto miss_dptr = raygen_dptr + sizeof(RayGenRecord);
  auto miss_sbt = MissRecord{};
  OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_miss_group, &miss_sbt));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(miss_dptr), &miss_sbt,
                        sizeof(MissRecord), cudaMemcpyHostToDevice));

  auto hit_sbts = std::vector<HitGroupRecord>{};
  for (auto i = 0u; i < triangle_offsets.size(); ++i) {
    auto rec = HitGroupRecord{};
    rec.data.object_idx = i;
    rec.data.triangle_idx = triangle_offsets.at(i);
    hit_sbts.emplace_back(rec);
    OPTIX_CHECK(
        optixSbtRecordPackHeader(state.radiance_hit_group, &hit_sbts.back()));
  }
  auto hit_dptr = miss_dptr + sizeof(MissRecord);
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(hit_dptr), hit_sbts.data(),
                        hit_sbts.size() * sizeof(HitGroupRecord),
                        cudaMemcpyHostToDevice));

  state.sbt.raygenRecord = raygen_dptr;
  state.sbt.missRecordBase = miss_dptr;
  state.sbt.missRecordStrideInBytes = sizeof(MissRecord);
  state.sbt.missRecordCount = 1;
  state.sbt.hitgroupRecordBase = hit_dptr;
  state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
  state.sbt.hitgroupRecordCount = hit_sbts.size();

  CUDA_CHECK(cudaStreamCreate(&state.stream));
  state.d_params = reinterpret_cast<Params *>(
      hit_dptr + hit_sbts.size() * sizeof(HitGroupRecord));
}

void SceneOptix::build_accel(
    PathTracerState &state,
    const std::vector<OptixBuildInput> &triangle_input) {
  int num_build_inputs = static_cast<int>(triangle_input.size());
  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
  accel_options.motionOptions.numKeys = 0;

  OptixAccelBufferSizes buffer_sizes;

  OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accel_options,
                                           triangle_input.data(),
                                           num_build_inputs, &buffer_sizes));
  auto d_temp_buffer = CUdeviceptr{0};
  auto output_buffer = CUdeviceptr{0};
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer),
                        buffer_sizes.tempSizeInBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&output_buffer),
                        buffer_sizes.outputSizeInBytes + 8));

  OptixAccelEmitDesc emit_property = {};
  emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emit_property.result = output_buffer + buffer_sizes.outputSizeInBytes;
  OPTIX_CHECK(optixAccelBuild(
      state.context, 0, &accel_options, triangle_input.data(), num_build_inputs,
      d_temp_buffer, buffer_sizes.tempSizeInBytes, output_buffer,
      buffer_sizes.outputSizeInBytes, &state.gas_handle, &emit_property, 1));

  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer)));

  size_t compact_size;
  CUDA_CHECK(cudaMemcpy(&compact_size,
                        reinterpret_cast<void *>(emit_property.result),
                        sizeof(size_t), cudaMemcpyDeviceToHost));

  if (compact_size < buffer_sizes.outputSizeInBytes) {
    auto compact_buffer = CUdeviceptr{0};
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&compact_buffer), compact_size));
    OPTIX_CHECK(optixAccelCompact(state.context,
                                  0, // CUDA stream
                                  state.gas_handle, compact_buffer,
                                  compact_size, &state.gas_handle));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(output_buffer)));
    output_buffer = compact_buffer;
  }

  // build CUDA stream
  state.params.handle = state.gas_handle;
  state.d_gas_output_buffer = output_buffer;
}

template <bool ad>
IntersectionOptix<ad> SceneOptix::hit(const Ray<ad> &ray,
                                      const Mask<ad> &valid) const {
  auto sz = enoki::slices(ray.direction);
  auto its = IntersectionOptix<ad>{};
  its.reserve(sz);

  enoki::cuda_eval();

  auto &params = m_accel->params;
  params.ray_o_x = ray.origin.x().data();
  params.ray_o_y = ray.origin.y().data();
  params.ray_o_z = ray.origin.z().data();

  params.ray_d_x = ray.direction.x().data();
  params.ray_d_y = ray.direction.y().data();
  params.ray_d_z = ray.direction.z().data();

  params.triangle_idx = its.triangle_idx.data();
  params.object_idx = its.object_idx.data();
  params.bary_u = its.uv.x().data();
  params.bary_v = its.uv.y().data();

  CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(m_accel->d_params),
                             &m_accel->params, sizeof(Params),
                             cudaMemcpyHostToDevice, m_accel->stream));

  OPTIX_CHECK(optixLaunch(m_accel->pipeline, m_accel->stream,
                          reinterpret_cast<CUdeviceptr>(m_accel->d_params),
                          sizeof(Params), &m_accel->sbt,
                          sz, // launch size
                          1,  // launch height
                          1   // launch depth
                          ));

  CUDA_SYNC_CHECK();
  return its;
}

template IntersectionOptixC SceneOptix::hit<false>(const RayC &ray,
                                                   const MaskC &valid) const;
template IntersectionOptixD SceneOptix::hit<true>(const RayD &ray,
                                                  const MaskD &valid) const;
