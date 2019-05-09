#ifndef GUARD_NEURALNET_SUBGRAPHOUTLINE_HPP
#define GUARD_NEURALNET_SUBGRAPHOUTLINE_HPP

#include <poponnx/transforms/transform.hpp>

namespace poponnx {

class SubgraphOutline : public Transform {
public:
  static std::size_t id();

  SubgraphOutline() : Transform() {}
  virtual ~SubgraphOutline() override {}

  virtual bool apply(Graph &graph) const override final;

  virtual std::size_t getId() const override final { return id(); }

  virtual std::string getName() const override final {
    return "SubgraphOutline";
  }
};

} // namespace poponnx

#endif
