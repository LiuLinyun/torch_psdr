#pragma once

#include "optix.h"

struct Params {
  const float* ray_o_x;
  const float* ray_o_y;
  const float* ray_o_z;

  const float* ray_d_x;
  const float* ray_d_y;
  const float* ray_d_z;

  int32_t* object_idx;
  int32_t* triangle_idx;

  float* bary_u;
  float* bary_v;

  OptixTraversableHandle handle;
};

struct RayGenData {};

struct MissData {};

struct HitGroupData {
  int32_t object_idx;
  int32_t triangle_idx;
};
