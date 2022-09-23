// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_STREAMINGMEMORY_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_STREAMINGMEMORY_HPP_

#include <cstddef>
#include <string>
#include <popart/transforms/transform.hpp>

#include "popart/names.hpp"

namespace popart {
class Graph;
class Op;

class StreamingMemory : public Transform {
public:
  static std::size_t id(int);

  StreamingMemory(int pass_) : Transform(), pass(pass_) {}
  ~StreamingMemory() override {}

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

private:
  int pass;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_STREAMINGMEMORY_HPP_
