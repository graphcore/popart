#ifndef GUARD_NEURALNET_BATCHNORM_HPP
#define GUARD_NEURALNET_BATCHNORM_HPP

#include <popart/op.hpp>

namespace popart {

class BatchNormOp : public Op {
public:
  BatchNormOp(const OperatorIdentifier &_opid,
              float _epsilon,
              float _momentum,
              int64_t _spatial,
              const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Inputs
  static InIndex getXInIndex() { return 0; }
  static InIndex getScaleInIndex() { return 1; }
  static InIndex getBInIndex() { return 2; }
  static InIndex getMeanInIndex() { return 3; }
  static InIndex getVarInIndex() { return 4; }

  // Ouputs
  static OutIndex getYOutIndex() { return 0; }
  static OutIndex getMeanOutIndex() { return 1; }
  static OutIndex getVarOutIndex() { return 2; }
  static OutIndex getSavedMeanOutIndex() { return 3; }
  static OutIndex getSavedVarOutIndex() { return 4; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  // Attributes
  float getEpsilon() const { return epsilon; }
  float getMomentum() const { return momentum; }
  int64_t getSpatial() const { return spatial; }

  bool isTraining() const {
    (void)isTest;
    return training;
  }

  void appendAttributes(OpSerialiserBase &) const override;

  bool isNorm() const override { return true; }

private:
  bool training = false;
  bool isTest;
  float epsilon;
  float momentum;
  int64_t spatial;
};

class BatchNormGradOp : public Op {
public:
  BatchNormGradOp(const BatchNormOp &);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getXInIndex() { return 0; }
  static InIndex getScaleInIndex() { return 1; }
  static InIndex getMeanInIndex() { return 2; }
  static InIndex getVarInIndex() { return 3; }
  static InIndex getYGradInIndex() { return 4; }

  static OutIndex getXOutIndex() { return 0; }
  static OutIndex getScaleOutIndex() { return 1; }
  static OutIndex getBOutIndex() { return 2; }

  float getEpsilon() const { return epsilon; }

  void appendAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

private:
  float epsilon;
  TensorInfo fwdInInfo, fwdScaleInInfo, fwdBInInfo;
};

} // namespace popart

#endif
