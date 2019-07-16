#ifndef GUARD_NEURALNET_GRADIENT_ACCUMULATION_HPP
#define GUARD_NEURALNET_GRADIENT_ACCUMULATION_HPP

#include <poponnx/op.hpp>
#include <poponnx/transforms/transform.hpp>

using VirtualGraphId = boost::optional<int64_t>;

namespace poponnx {

class GradientAccumulation : public Transform {
public:
  static std::size_t id();

  GradientAccumulation() : Transform() {}
  virtual ~GradientAccumulation() override {}

  bool apply(Graph &graph) const final;

  std::size_t getId() const final { return id(); }

  std::string getName() const final { return "GradientAccumulation"; }
};

} // namespace poponnx

#endif
