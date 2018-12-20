#ifndef GUARD_NEURALNET_SOFTMAXXXXX_HPP
#define GUARD_NEURALNET_SOFTMAXXXXX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class SoftmaxOp;
class SoftmaxGradOp;
class SoftmaxGradDirectOp;

namespace popx {

class SoftmaxOpx : public Opx {
public:
  SoftmaxOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class SoftmaxGradOpx : public Opx {
public:
  SoftmaxGradOpx(Op *, Devicex *);
};

class SoftmaxGradDirectOpx : public Opx {
public:
  SoftmaxGradDirectOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
