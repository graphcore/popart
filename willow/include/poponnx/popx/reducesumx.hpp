#ifndef GUARD_NEURALNET_REDUCESUMX_HPP
#define GUARD_NEURALNET_REDUCESUMX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace willow {

class ReduceSumOp;

namespace popx {

class ReduceSumOpx : public Opx {
public:
  ReduceSumOpx(Op *, Devicex *);
  virtual void grow() const override final;
};

class ReduceSumGradOpx : public Opx {
public:
  ReduceSumGradOpx(Op *, Devicex *);
  virtual void grow() const override final;
};

} // namespace popx
} // namespace willow

#endif
