#ifndef GUARD_NEURALNET_SOFTMAXXXXX_HPP
#define GUARD_NEURALNET_SOFTMAXXXXX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace willow {

class SoftmaxOp;
class SoftmaxGradOp;
class SoftmaxGradDirectOp;

namespace popx {

class SoftmaxOpx : public Opx {
public:
  SoftmaxOpx(Op *, Devicex *);
  SoftmaxOp *getSoftmaxOp() const;
  void grow(poplar::program::Sequence &) const override final;
};

class SoftmaxGradOpx : public Opx {
public:
  SoftmaxGradOpx(Op *, Devicex *);
  SoftmaxGradOp *getSoftmaxGradOp() const;
};

class SoftmaxGradDirectOpx : public Opx {
public:
  SoftmaxGradDirectOpx(Op *, Devicex *);
  SoftmaxGradDirectOp *getSoftmaxGradDirectOp() const;
  void grow(poplar::program::Sequence &) const override final;
};

} // namespace popx
} // namespace willow

#endif
