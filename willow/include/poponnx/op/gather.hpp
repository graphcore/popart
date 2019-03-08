#ifndef GUARD_NEURALNET_GATHER_HPP
#define GUARD_NEURALNET_GATHER_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class GatherOp : public Op {
public:
  GatherOp(const OperatorIdentifier &_opid,
           int64_t axis_,
           const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Which axis to gather on.
  int64_t getAxis() const;

  static InIndex dataInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static InIndex outIndex() { return 0; }

  void appendAttributes(std::stringstream &ss,
                        const std::string &tab) const override;

  bool canBeReplacedByIdentity() override;

private:
  int64_t axis = 0;
};

class GatherGradOp : public Op {
public:
  GatherGradOp(const GatherOp &op, int64_t axis);

  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  // Which axis to gather on.
  int64_t getAxis() const;

  static InIndex gradInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static InIndex gradOutIndex() { return 0; }

  void appendAttributes(std::stringstream &ss,
                        const std::string &tab) const override;

private:
  int64_t axis;
  TensorInfo fwdDataInfo;
};

} // namespace poponnx

#endif
