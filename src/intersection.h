#pragma once

#include "src/types.h"

class Material;
class Light;

template <typename Value>
struct Intersection_ {
  Vec3<Value> point;
  Vec3<Value> normal;   // always NdotV >= 0
  Vec3<Value> tangent;  // dP/du
  Vec3<Value> wi;       // from current point to ray origin
  Value squared_dist;
  Vec2<Value> uv;  // texture space coordinates
  Value jacobi;
  Type<Material*, enoki::is_diff_array_v<Value>> material;
  Type<Light*, enoki::is_diff_array_v<Value>> light;
  enoki::mask_t<Value> is_front;
  enoki::mask_t<Value> is_light;
  enoki::mask_t<Value> valid;

  ENOKI_STRUCT(Intersection_, point, normal, tangent, wi, squared_dist, uv, jacobi,
               material, light, is_front, is_light, valid)
};
ENOKI_STRUCT_SUPPORT(Intersection_, point, normal, tangent, wi, squared_dist, uv,
                     jacobi, material, light, is_front, is_light, valid)

template <bool ad>
using Intersection = Intersection_<Float<ad>>;

using IntersectionC = Intersection<false>;
using IntersectionD = Intersection<true>;
