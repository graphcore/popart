#ifndef GUARD_NEURALNET_MEAN_HPP
#define GUARD_NEURALNET_MEAN_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class MeanOp : public Op {
public:
  MeanOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Mean has a variable number of inputs
  static OutIndex getOutIndex() { return 0; }

  bool canBeReplacedByIdentity() override;
};

// A MeanGradOp will be created for each input to MeanOp i.e. it will compute
// the gradient of a single input argument
class MeanGradOp : public Op {
public:
  MeanGradOp(const MeanOp &, InIndex index);

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  unsigned getNumFwdOpInputs() { return numFwdOpInputs; }

private:
  InIndex fwdIndex;
  unsigned numFwdOpInputs;
  std::map<int, int> gradOutToNonGradInInfo;
  std::vector<GradInOutMapper> gradInputInfoVec;
};

} // namespace poponnx

#endif
