#ifndef GUARD_NEURALNET_ADDX_HPP
#define GUARD_NEURALNET_ADDX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace willow {

class AddOp;
class AddGradOp;

namespace popx {

class AddOpx : public Opx {
public:
  AddOpx(Op *, Devicex *);
  AddOp *getAddOp() const;
  virtual void grow() const override final;
};

class AddGradOpx : public Opx {
public:
  AddGradOpx(Op *, Devicex *);
  AddGradOp *getAddGradOp() const;
};

} // namespace popx
} // namespace willow

#endif
