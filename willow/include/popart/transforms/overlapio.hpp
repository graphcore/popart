// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OVERLAPIO_HPP
#define GUARD_NEURALNET_OVERLAPIO_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

class OverlapIO : public Transform {
public:
  static std::size_t id();

  OverlapIO() : Transform() {}
  virtual ~OverlapIO() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "OverlapIO"; }
};

} // namespace popart

#endif
