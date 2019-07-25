#ifndef GUARD_NEURALNET_PRUNE_HPP
#define GUARD_NEURALNET_PRUNE_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

class Prune : public Transform {
public:
  static std::size_t id();

  Prune() : Transform() {}
  virtual ~Prune() override {}

  virtual bool apply(Graph &graph) const override final;

  virtual std::size_t getId() const override final { return id(); }

  virtual std::string getName() const override final { return "Prune"; }
};

} // namespace popart

#endif
