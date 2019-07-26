#ifndef GUARD_NEURALNET_GRADIENT_ACCUMULATION_HPP
#define GUARD_NEURALNET_GRADIENT_ACCUMULATION_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

using VirtualGraphId = boost::optional<int64_t>;

namespace popart {

class GradientAccumulation : public Transform {
public:
  static std::size_t id();

  GradientAccumulation() : Transform() {}
  virtual ~GradientAccumulation() override {}

  bool apply(Graph &graph) const final;

  std::size_t getId() const final { return id(); }

  std::string getName() const final { return "GradientAccumulation"; }
};

} // namespace popart

#endif
