#ifndef GUARD_NEURALNET_ADDX_HPP
#define GUARD_NEURALNET_ADDX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/identityx.hpp>
#include <poponnx/popx/opx.hpp>

namespace willow {

class AddOp;

namespace popx {

class AddOpx : public Opx {
public:
  AddOpx(Op *, Devicex *);
  AddOp *getAddOp() const;
  virtual void grow(poplar::program::Sequence &) const override final;
};

// TODO (task T5432) should inherit from ReduceSum when we have numpy
// broadcasting
class AddArg0GradOpx : public IdentityOpx {
public:
  AddArg0GradOpx(Op *, Devicex *);
};

// TODO (task T5432) should inherit from ReduceSum when we have numpy
// broadcasting
class AddArg1GradOpx : public IdentityOpx {
public:
  AddArg1GradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace willow

#endif
