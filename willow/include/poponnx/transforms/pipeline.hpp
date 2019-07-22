#ifndef GUARD_NEURALNET_PIPELINE_HPP
#define GUARD_NEURALNET_PIPELINE_HPP

#include <poponnx/op/restore.hpp>
#include <poponnx/transforms/transform.hpp>

namespace poponnx {

class Pipeline : public Transform {
public:
  static std::size_t id();

  Pipeline() : Transform() {}
  virtual ~Pipeline() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "Pipeline"; }

private:
  int64_t getVirtualGraphIdOrSourceIpu(Op *op) const;

  RestoreOp *addNewRestoreOp(Graph &graph) const;
  RestoreOp *addNewRestoreInplaceOp(Graph &graph) const;
};

} // namespace poponnx

#endif
