#ifndef GUARD_NEURALNET_CONV_HPP
#define GUARD_NEURALNET_CONV_HPP

#include <poponnx/op/receptive.hpp>

namespace poponnx {

class ConvOp : public HasReceptiveFieldOp {
public:
  ConvOp(const OperatorIdentifier &_opid,
         Ir *_ir,
         const std::string &name = "",
         const Attributes &_attr = {});
  int64_t nOutChans;
  int64_t group;
  bool cacheOperation = true;
  // convenience functions:
  const Tensor *dataIn() const;
  const Tensor *weightsIn() const;
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  int64_t getNOutChans() const final;

  // from github.com/onnx/onnx/blob/master/docs/Operators.md#Conv :
  // "data" at index 0, "weights" at index 1, "bias" as index 2.
  // poponnx's ConvOp does not support bias, but bias can be used when
  // ConvBiasPattern is used.
  static InIndex getDataInIndex() { return 0; }
  static InIndex getWeightsInIndex() { return 1; }
  static InIndex getBiasInIndex() { return 2; }
  static OutIndex getOutIndex() { return 0; }

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
  static InIndex getGradConvolvedInIndex() { return 0; }

  // The input index where the input to the
  // convolution (ConvOp) is inserted to this Op
  static InIndex getPreConvolvedInIndex() { return 1; }

  static OutIndex getOutIndex() { return 0; }

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
  static InIndex getWeightsInIndex() { return 0; }
  // The input index where the gradient of the output is inserted
  static InIndex getGradConvolvedInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

private:
  std::unique_ptr<Op> cloneOfCreator;
  TensorInfo dataInfo;
};

} // namespace poponnx

#endif
