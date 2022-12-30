#pragma once

#include "src/scene.h"

class Integrator {
 public:
  virtual ~Integrator() = default;
  virtual std::vector<SpectrumC> renderC(const Scene& scene) const = 0;
  virtual std::vector<SpectrumD> renderD(const Scene& scene) const = 0;
};

