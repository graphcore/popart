#ifndef GUARD_NEURALNET_CONVX_HPP
#define GUARD_NEURALNET_CONVX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class ConvOp;
class ConvWeightsGradOp;
class ConvDataGradOp;

namespace popx {

class ConvOpx : public Opx {
public:
  ConvOpx(Op *);
  ConvOp *getConvOp() const;
  virtual poplar::Tensor createInput(int index) const override final;
  virtual bool canCreateInput(int index) const override final;
};

class ConvDataGradOpx : public Opx {
public:
  ConvDataGradOpx(Op *);
  ConvDataGradOp *getConvDataGradOp() const;
};

class ConvWeightsGradOpx : public Opx {
public:
  ConvWeightsGradOpx(Op *);
  ConvWeightsGradOp *getConvWeightsGradOp() const;
};

} // namespace popx
} // namespace willow

#endif
