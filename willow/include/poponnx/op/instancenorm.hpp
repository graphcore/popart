#ifndef GUARD_NEURALNET_INSTANCENORM_HPP
#define GUARD_NEURALNET_INSTANCENORM_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class InstanceNormOp : public Op {
public:
  InstanceNormOp(const OperatorIdentifier &_opid,
                 float _epsilon,
                 const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Input's
  static InIndex getInputInIndex() { return 0; }
  static InIndex getScaleInIndex() { return 1; }
  static InIndex getBInIndex() { return 2; }

  // Ouput's
  static OutIndex getOutIndex() { return 0; }
  static OutIndex getMeanOutIndex() { return 1; }
  static OutIndex getInvStdDevOutIndex() { return 2; }

  // Attributes
  float getEpsilon() const { return epsilon; }

  void appendAttributes(std::stringstream &ss,
                        const std::string &tab) const override;

private:
  float epsilon;
};

class InstanceNormGradOp : public Op {
public:
  InstanceNormGradOp(const InstanceNormOp &fwd_op);

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getInputInIndex() { return 0; }
  static InIndex getScaleInIndex() { return 1; }
  static InIndex getOutGradInIndex() { return 2; }
  static InIndex getMeanInIndex() { return 3; }
  static InIndex getInvStdDevInIndex() { return 4; }

  static OutIndex getInputOutIndex() { return 0; }
  static OutIndex getScaleOutIndex() { return 1; }
  static OutIndex getBOutIndex() { return 2; }
};

} // namespace poponnx

#endif
