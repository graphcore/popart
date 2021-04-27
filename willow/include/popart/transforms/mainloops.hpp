// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MAINLOOPS_HPP
#define GUARD_NEURALNET_MAINLOOPS_HPP

#include <popart/graph.hpp>
#include <popart/op/loop.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

class MainLoops : public Transform {
public:
  static std::size_t id();

  MainLoops() : Transform() {}
  virtual ~MainLoops() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "MainLoops"; }
};

} // namespace popart

#endif
