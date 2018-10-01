#ifndef GUARD_NEURALNET_ADD_HPP
#define GUARD_NEURALNET_ADD_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet {

class AddOp : public Op {
public:
  AddOp(const onnx::NodeProto &node, Graph *pgraph);
  virtual std::unique_ptr<Op> clone() const override final;
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual void setup() override final;
};

class AddGradOp : public GradOp {

public:
  AddGradOp(AddOp *);
  virtual Op *getNonGradCreator() const override final;
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;

private:
  std::vector<GradInOutMapper> createAddGradInfo() const;
  std::map<int, int> createAddGradOutToIn() const;
  AddOp *addOp;
};

} // namespace neuralnet

#endif
