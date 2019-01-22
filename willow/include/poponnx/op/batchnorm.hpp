#ifndef GUARD_NEURALNET_BATCHNORM_HPP
#define GUARD_NEURALNET_BATCHNORM_HPP

#include <poponnx/op.hpp>

namespace poponnx {

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

  // Input's
  static InIndex getXInIndex() { return 0; }
  static InIndex getScaleInIndex() { return 1; }
  static InIndex getBInIndex() { return 2; }
  static InIndex getMeanInIndex() { return 3; }
  static InIndex getVarInIndex() { return 4; }

  // Ouput's
  static OutIndex getYOutIndex() { return 0; }
  static OutIndex getMeanOutIndex() { return 1; }
  static OutIndex getVarOutIndex() { return 2; }
  static OutIndex getSavedMeanOutIndex() { return 3; }
  static OutIndex getSavedVarOutIndex() { return 4; }

  // Attributes
  float getEpsilon() const { return epsilon; }
  float getMomentum() const { return momentum; }
  int64_t getSpatial() const { return spatial; }

  bool isTraining() const { return training; }

  void appendAttributes(std::stringstream &ss,
                        const std::string &tab) const override;

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
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  const BatchNormOp &getFwdOp() { return fwdOp; }

  static InIndex getXInIndex() { return 0; }
  static InIndex getScaleInIndex() { return 1; }
  static InIndex getMeanInIndex() { return 2; }
  static InIndex getVarInIndex() { return 3; }
  static InIndex getYGradInIndex() { return 4; }

  static OutIndex getXOutIndex() { return 0; }
  static OutIndex getScaleOutIndex() { return 1; }
  static OutIndex getBOutIndex() { return 2; }

private:
  const BatchNormOp &fwdOp;
  // TensorInfo forward_op_arg_info;
};

} // namespace poponnx

#endif
