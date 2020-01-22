#ifndef GUARD_NEURALNET_MERGEDUPLICATEOPS_HPP
#define GUARD_NEURALNET_MERGEDUPLICATEOPS_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

class MergeDuplicateOps : public Transform {
public:
  static std::size_t id();

  MergeDuplicateOps() : Transform() {}
  virtual ~MergeDuplicateOps() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "MergeDuplicateOps"; }
};

} // namespace popart

#endif
