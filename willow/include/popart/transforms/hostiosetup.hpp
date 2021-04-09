// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_HOSTIOSETUP_HPP
#define GUARD_NEURALNET_HOSTIOSETUP_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

class HostIOSetup : public Transform {
public:
  static std::size_t id();

  HostIOSetup() : Transform() {}
  virtual ~HostIOSetup() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "HostIOSetup"; }

private:
  void setupMainGraphHostLoadOps(HostStreamId, Tensor *, Graph *) const;
  void setupMainGraphHostStoreOps(HostStreamId, Tensor *, Graph *) const;
};

} // namespace popart

#endif
