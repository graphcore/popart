#ifndef GUARD_NEURALNET_EXPLICITRECOMPUTE_HPP
#define GUARD_NEURALNET_EXPLICITRECOMPUTE_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

class ExplicitRecompute : public Transform {
public:
  static std::size_t id();

  ExplicitRecompute() : Transform() {}
  virtual ~ExplicitRecompute() override {}

  virtual bool apply(Graph &graph) const override final;

  virtual std::size_t getId() const override final { return id(); }

  virtual std::string getName() const override final {
    return "ExplicitRecompute";
  }
};

} // namespace popart

#endif
