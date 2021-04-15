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

  // Apply autodiff to entire IR.
  virtual bool apply(Ir &ir) const final;

  // This method is not implemented at the moment.
  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "Autodiff"; }
};

} // namespace popart

#endif
