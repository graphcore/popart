// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_AUTODIFF_HPP
#define GUARD_NEURALNET_AUTODIFF_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

// Forward declarations.
class AutodiffIrInterface;

class Autodiff : public Transform {
public:
  static std::size_t id();

  Autodiff() : Transform() {}
  virtual ~Autodiff() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "Autodiff"; }
};

} // namespace popart

#endif
