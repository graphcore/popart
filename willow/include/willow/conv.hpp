#ifndef GUARD_NEURALNET_CONV_HPP
#define GUARD_NEURALNET_CONV_HPP

#include <willow/receptive.hpp>

namespace willow {

class ConvOp : public HasReceptiveFieldOp {
public:
  ConvOp(const onnx::NodeProto &node, Ir *pir);
  int64_t nOutChans;
  int64_t group;
  // from github.com/onnx/onnx/blob/master/docs/Operators.md#Conv :
  // data at index 0, weights at index 1.
  // willow's ConvOp does not support bias.
  int dataInIndex() const { return 0; }
  int weightsInIndex() const { return 1; }
  // convenience functions:
  const Tensor *dataIn() const;
  const Tensor *weightsIn() const;
  virtual std::unique_ptr<Op> clone() const override final;
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual int64_t getNOutChans() const override final;

private:
  virtual void setup0() override final;
  virtual void setSpatialK() override final;
};

class ConvWeightsGradOp : public GradOp {
public:
  ConvWeightsGradOp(ConvOp *);
  virtual Op *getNonGradCreator() const override final;
  // equivalent of getNonGradCreator, but no downcasting
  ConvOp *getConvOp() const;
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;

private:
  std::vector<GradInOutMapper> createConvWeightsGradInfo() const;
  std::map<int, int> createConvWeightsGradOutToIn() const;
  ConvOp *convOp;
};

class ConvDataGradOp : public GradOp {
public:
  ConvDataGradOp(ConvOp *);
  virtual Op *getNonGradCreator() const override final;
  // equivalent of getNonGradCreator, but no downcasting
  ConvOp *getConvOp() const;
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;

  // The input index where the weight tensor is inserted
  int getWeightsIn() const;
  // The input index where the gradient of the output is inserted
  int getGradConvolvedIn() const;

private:
  std::vector<GradInOutMapper> createConvDataGradInfo() const;
  std::map<int, int> createConvDataGradOutToIn() const;
  ConvOp *convOp;
};

} // namespace willow

#endif
