// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_INFERPIPELINESTAGES_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_INFERPIPELINESTAGES_HPP_

#include <cstddef>
#include <string>
#include <popart/transforms/transform.hpp>

namespace popart {
class Graph;

class InferPipelineStages : public Transform {
public:
  static std::size_t id();

  InferPipelineStages() : Transform() {}
  virtual ~InferPipelineStages() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "InferPipelineStages"; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_INFERPIPELINESTAGES_HPP_
