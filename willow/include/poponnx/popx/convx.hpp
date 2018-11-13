#ifndef GUARD_NEURALNET_CONVX_HPP
#define GUARD_NEURALNET_CONVX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/enigma.hpp>
#include <poponnx/popx/opx.hpp>

#include <poplin/Convolution.hpp>

namespace willow {

class ConvOp;
class ConvWeightsGradOp;
class ConvDataGradOp;

namespace popx {

poplin::ConvParams getFwdConvParams(const ConvOp *cOp);
poplin::ConvParams getDataGradParams(const ConvDataGradOp *convDataGradOp);

class ConvOpx : public Opx {
public:
  ConvOpx(Op *, Devicex *);
  ConvOp *getConvOp() const;
  virtual poplar::Tensor createInput(int index) const override final;
  virtual bool canCreateInput(int index) const override final;
  virtual bool createsEquiv(int, Opx *, int) const override final;
  virtual std::vector<TensorId>
  mustExistBeforeCreate(int index0) const override final;
  const poplin::ConvParams &getParams() const;
  virtual void grow() const override final;

private:
  poplin::ConvParams fwdParams;
};

class ConvDataGradOpx : public Opx {
public:
  ConvDataGradOpx(Op *, Devicex *);
  ConvDataGradOp *getConvDataGradOp() const;
  virtual void grow() const override final;
  const poplin::ConvParams &getParams() const;

private:
  poplin::ConvParams dataGradParams;
};

class ConvWeightsGradOpx : public Opx {
public:
  ConvWeightsGradOpx(Op *, Devicex *);
  ConvWeightsGradOp *getConvWeightsGradOp() const;
  virtual void grow() const override final;
};

} // namespace popx
} // namespace willow

#endif
