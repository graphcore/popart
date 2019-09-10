#ifndef GUARD_NEURALNET_PIPELINE_HPP
#define GUARD_NEURALNET_PIPELINE_HPP

#include <popart/op/restore.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

class Pipeline : public Transform {
public:
  static std::size_t id();

  Pipeline() : Transform() {}
  virtual ~Pipeline() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "Pipeline"; }

private:
  RestoreOp *addNewRestoreOp(Graph &graph) const;
  RestoreInplaceOp *addNewRestoreInplaceOp(Graph &graph) const;
};

} // namespace popart

#endif
