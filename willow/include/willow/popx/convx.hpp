#ifndef GUARD_NEURALNET_CONVX_HPP
#define GUARD_NEURALNET_CONVX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class ConvOp;

namespace popx {

class ConvOpx : public Opx {
public:
  ConvOpx(Op *);
  ConvOp *getConvOp() const;
  virtual poplar::Tensor createInput(int index) const override final;
  virtual bool canCreateInput(int index) const override final;
};

} // namespace popx
} // namespace willow

#endif
