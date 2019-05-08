#ifndef GUARD_NEURALNET_NLL_HPP
#define GUARD_NEURALNET_NLL_HPP

#include <poponnx/op.hpp>
#include <poponnx/op/loss.hpp>

namespace poponnx {

class NllLoss : public Loss {
public:
  NllLoss(TensorId probs, TensorId label, TensorId output);
  NllLoss(TensorId probs, TensorId label, TensorId output, int ignoreIndex);

  // label is the only streamed input tensor to this loss
  std::vector<TensorId> getStreamTensorNames() const final;
  std::unique_ptr<Op> getOp(const Op::Settings &settings_) const final;
  const OperatorIdentifier &op_type() const final;

  static InIndex getProbsInIndex() { return 0; }
  static InIndex getLabelInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  TensorId probsTensorId() const;
  TensorId labelTensorId() const;
  std::unique_ptr<Loss> clone() const final;
  bool hasIgnoreIndex() const { return hasIgnoreIndex_; }
  int getIgnoreIndex() const { return ignoreIndex_; }

private:
  // Specifies a target value that is masked when calculating the loss and
  // input gradient
  int ignoreIndex_;

  // Has ignoreIndex_ been set?
  bool hasIgnoreIndex_ = false;
};

class NllOp : public LossOp {
public:
  NllOp(const OperatorIdentifier &_opid,
        const NllLoss *nllloss,
        const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static OutIndex getOutIndex() { return 0; }

  const NllLoss *nlll() const;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  const NllLoss *nllloss_;
};

class NllGradOp : public Op {
public:
  NllGradOp(const NllOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static OutIndex getOutIndex() { return 0; }

  const NllLoss *nlll() const;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  const NllLoss *nllloss_;
};

} // namespace poponnx

#endif
