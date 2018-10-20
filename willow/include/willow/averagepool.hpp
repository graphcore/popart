#ifndef GUARD_NEURALNET_AVERAGEPOOL_HPP
#define GUARD_NEURALNET_AVERAGEPOOL_HPP

#include <willow/ir.hpp>
#include <willow/names.hpp>
#include <willow/receptive.hpp>

namespace willow {

// c++ note : the conditions are suitable here
// for the compiler to generate defaults for
// "the 3": destructor, copy constructor, assigment op.
class AveragePoolOp : public HasReceptiveFieldOp {
public:
  AveragePoolOp(const onnx::NodeProto &node, Ir *pir);
  virtual std::unique_ptr<Op> clone() const override final;
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  int64_t getNOutChans() const override final;

private:
  virtual void setup0() override final;
  virtual void setSpatialK() override final;
};

class AveragePoolGradOp : public GradOp {
public:
  AveragePoolGradOp(AveragePoolOp *);
  virtual Op *getNonGradCreator() const override final;
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  void setup() override final;

private:
  std::vector<GradInOutMapper> createAveragePoolGradInfo() const;
  std::map<int, int> createAveragePoolGradOutToIn() const;
  AveragePoolOp *averagePoolOp;
};

} // namespace willow

#endif
