#include "params.h"
#include <cmath>
#include <limits>

// compile: nvcc --ptx -arch=native device_programs.cu -I./ -I/opt/optix/include

using Real = float;
constexpr auto REAL_EPS = Real{1e-3};
constexpr auto REAL_INF = std::numeric_limits<Real>::infinity();

extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__fn() {
  const auto tid = optixGetLaunchIndex().x;
  optixTrace(params.handle,
             make_float3(params.ray_o_x[tid], params.ray_o_y[tid],
                         params.ray_o_z[tid]),
             make_float3(params.ray_d_x[tid], params.ray_d_y[tid],
                         params.ray_d_z[tid]),
             REAL_EPS,                // tmin
             REAL_INF,                // tmax
             0.0f,                    // ray time
             OptixVisibilityMask(1),  // visibilityMask
             OPTIX_RAY_FLAG_NONE,     // rayFlags
             0,                       // SBToffset
             1,                       // SBTstride
             0                        // missSBTIndex
  );
}

extern "C" __global__ void __miss__fn() {
  const auto tid = optixGetLaunchIndex().x;
  params.triangle_idx[tid] = -1;
  params.object_idx[tid] = -1;
  params.bary_u[tid] = -1.0f;
  params.bary_v[tid] = -1.0f;
}

extern "C" __global__ void __closesthit__fn() {
  auto rt_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const auto tid = optixGetLaunchIndex().x;
  params.triangle_idx[tid] = optixGetPrimitiveIndex() + rt_data->triangle_idx;
  params.object_idx[tid] = rt_data->object_idx;
  const auto& uv = optixGetTriangleBarycentrics();
  params.bary_u[tid] = uv.x;
  params.bary_v[tid] = uv.y;
}
