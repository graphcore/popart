#ifndef GUARD_NEURALNET_SOFTMAXXXXX_HPP
#define GUARD_NEURALNET_SOFTMAXXXXX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

class SoftmaxOp;
class SoftmaxGradOp;
class SoftmaxGradDirectOp;

namespace popx {

class SoftmaxOpx : public ElementWiseUnaryOpx {
public:
  SoftmaxOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  static poplar::Tensor coerceTo2D(const poplar::Tensor &t, int64_t axis);
};

// compute dL/dv from v and dp, where p = softmax(v)
class SoftmaxGradOpx : public ElementWiseUnaryOpx {
public:
  SoftmaxGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

// compute dL/dv from lab and p, where p = softmax(v), L = nll(p, lab)
class SoftmaxGradDirectOpx : public ElementWiseUnaryOpx {
public:
  SoftmaxGradDirectOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
