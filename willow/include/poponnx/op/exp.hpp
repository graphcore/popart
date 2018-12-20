#ifndef GUARD_NEURALNET_EXP_HPP
#define GUARD_NEURALNET_EXP_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class ExpOp : public Op {
public:
  ExpOp(const OperatorIdentifier &_opid,
        Ir *_ir,
        const std::string &name = "",
        const Attributes &_attr = {});
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

class ExpGradOp : public Op {
public:
  ExpGradOp(ExpOp *fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdOutInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace poponnx

#endif
