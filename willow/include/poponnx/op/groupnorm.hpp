#ifndef GUARD_NEURALNET_GROUPNORM_HPP
#define GUARD_NEURALNET_GROUPNORM_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class GroupNormOp : public Op {
public:
  GroupNormOp(const OperatorIdentifier &opid_,
              int64_t num_groups_,
              float epsilon_,
              const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Input's
  static InIndex getXInIndex() { return 0; }
  static InIndex getScaleInIndex() { return 1; }
  static InIndex getBInIndex() { return 2; }

  // Ouput's
  static OutIndex getYOutIndex() { return 0; }
  static OutIndex getMeanOutIndex() { return 1; }
  static OutIndex getVarOutIndex() { return 2; }

  // Attributes
  float getEpsilon() const { return epsilon; }
  int64_t getNumGroups() const { return num_groups; }

  void appendAttributes(OpSerialiserBase &) const override;

private:
  int64_t num_groups;
  float epsilon;
};

class GroupNormGradOp : public Op {
public:
  GroupNormGradOp(const GroupNormOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getXInIndex() { return 0; }
  static InIndex getScaleInIndex() { return 1; }
  static InIndex getMeanInIndex() { return 2; }
  static InIndex getVarInIndex() { return 3; }
  static InIndex getYGradInIndex() { return 4; }

  static OutIndex getXGradOutIndex() { return 0; }
  static OutIndex getScaleOutIndex() { return 1; }
  static OutIndex getBOutIndex() { return 2; }

  float getEpsilon() const { return epsilon; }

private:
  float epsilon;
  TensorInfo fwdInInfo, fwdScaleInInfo, fwdBInInfo;
};

} // namespace poponnx

#endif
