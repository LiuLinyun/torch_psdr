#pragma once

#include "src/types.h"

template <bool ad>
struct Ray {
  Ray() = delete;
  Ray(const Vec3f<ad>& origin_, const Vec3f<ad>& direction_)
      : origin{origin_}, direction{direction_} {}

  Vec3f<ad> at(const Float<ad>& t) const { return origin + direction * t; }

  Vec3f<ad> origin;
  Vec3f<ad> direction;
};

using RayC = Ray<false>;
using RayD = Ray<true>;
