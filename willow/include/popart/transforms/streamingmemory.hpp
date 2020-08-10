// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_STREAMINGMEMORY_HPP
#define GUARD_NEURALNET_STREAMINGMEMORY_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

class StreamingMemory : public Transform {
public:
  static std::size_t id(int);

  StreamingMemory(int pass_) : Transform(), pass(pass_) {}
  virtual ~StreamingMemory() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(pass); }

  virtual std::string getName() const final {
    return "StreamingMemory " + std::to_string(pass);
  }

private:
  TensorId generateRemoteArgTensorId(TensorId tid, VGraphId vgid) const;

  float costFn(Op *op) const;

  void verifyPlacementConsistency(const Op *op, unsigned num_stages) const;

  void verifyExecutionPhases(Graph &graph) const;

  void getModifiersInPhase(ExecutionPhase phase,
                           Tensor *t,
                           std::vector<Op *> &modifyingConsumerOps) const;

  void
  getAliasedModifiersInPhase(Graph &graph,
                             ExecutionPhase phase,
                             Tensor *t,
                             std::vector<Op *> &modifyingConsumerOps) const;

private:
  int pass;
};

} // namespace popart

#endif
