#ifndef GUARD_NEURALNET_CONV_HPP
#define GUARD_NEURALNET_CONV_HPP

#include <poponnx/op/receptive.hpp>

namespace willow {

class ConvOp : public HasReceptiveFieldOp {
public:
  ConvOp(const onnx::NodeProto &node, Ir *pir);
  int64_t nOutChans;
  int64_t group;
  // convenience functions:
  const Tensor *dataIn() const;
  const Tensor *weightsIn() const;
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  int64_t getNOutChans() const final;

  // from github.com/onnx/onnx/blob/master/docs/Operators.md#Conv :
  // "data" at index 0, "weights" at index 1, "bias" as index 2.
  // willow's ConvOp does not support bias, but bias can be used when
  // ConvBiasPattern is used.
  static int dataInIndex();
  static int weightsInIndex();
  static int biasInIndex();

private:
  void setup0() final;
  void setSpatialK() final;
};

class ConvWeightsGradOp : public Op {
public:
  ConvWeightsGradOp(ConvOp *);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  // The input index where the gradient of
  // the output of convolution is inserted to this Op
  int getGradConvolvedIn() const;

  // The input index where the input to the
  // convolution (ConvOp) is inserted to this Op
  int getPreConvolvedIn() const;
  const ConvOp *getCloneOfCreator() const;

private:
  std::unique_ptr<Op> cloneOfCreator;
  TensorInfo weightsInfo;
};

class ConvDataGradOp : public Op {
public:
  ConvDataGradOp(ConvOp *);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  const ConvOp *getCloneOfCreator() const;

  // The input index where the weight tensor is inserted
  int getWeightsIn() const;
  // The input index where the gradient of the output is inserted
  int getGradConvolvedIn() const;

private:
  std::unique_ptr<Op> cloneOfCreator;
  TensorInfo dataInfo;
};

} // namespace willow

#endif
