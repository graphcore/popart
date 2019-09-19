#ifndef GUARD_NEURALNET_INFERPIPELINESTAGES_HPP
#define GUARD_NEURALNET_INFERPIPELINESTAGES_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

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

#endif
