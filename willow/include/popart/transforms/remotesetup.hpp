// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REMOTESETUP_HPP
#define GUARD_NEURALNET_REMOTESETUP_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

class RemoteSetup : public Transform {
public:
  static std::size_t id();

  RemoteSetup() : Transform() {}
  virtual ~RemoteSetup() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "RemoteSetup"; }
};

} // namespace popart

#endif
