#ifndef GUARD_NEURALNET_CONV_HPP
#define GUARD_NEURALNET_CONV_HPP

#include <neuralnet/receptive.hpp>

namespace neuralnet {

class ConvOp : public HasReceptiveFieldOp {
public:
  ConvOp(const onnx::NodeProto &node, Graph *pgraph);
  int64_t nOutChans;
  int64_t group;
  // from github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
  int dataInIndex() { return 0; }
  int weightsInIndex() { return 1; }

  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;

private:
  virtual int64_t getNOutChans() const override final;
  virtual void setup0() override final;
  virtual void setSpatial() override final;
};

class ConvWeightsGradOp : public GradOp {
public:
  ConvWeightsGradOp(ConvOp *);
  virtual Op *getNonGradCreator() const override final;
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
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;
  virtual void imposeTopoCons() override final;

private:
  std::vector<GradInOutMapper> createConvDataGradInfo() const;
  std::map<int, int> createConvDataGradOutToIn() const;
  ConvOp *convOp;
};

} // namespace neuralnet

#endif
