
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RANDOMSETUP_HPP
#define GUARD_NEURALNET_RANDOMSETUP_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

class RandomSetup : public Transform {
public:
  static std::size_t id();

  RandomSetup() : Transform() {}
  virtual ~RandomSetup() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "RandomSetup"; }

  static bool hasRandomSeed(const Ir &ir);
  static bool requiresRandomSeed(const Ir &ir);
  static TensorId getStreamedSeedTensorId();

protected:
  static bool hasRandomOps(const Ir &ir);
  static std::tuple<VGraphId, PipelineStage> getGroupKey(const Op *op);
};

} // namespace popart

#endif
