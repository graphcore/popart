#ifndef GUARD_NEURALNET_PRUNE_HPP
#define GUARD_NEURALNET_PRUNE_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

class Prune : public Transform {
public:
  static std::size_t id();

  Prune() : Transform() {}
  virtual ~Prune() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "Prune"; }
};

} // namespace popart

#endif
