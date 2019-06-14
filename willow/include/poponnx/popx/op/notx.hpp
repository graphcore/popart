#ifndef GUARD_NEURALNET_NOTX_HPP
#define GUARD_NEURALNET_NOTX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

class NotOp;

namespace popx {

class NotOpx : public ElementWiseUnaryOpx {
public:
  NotOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
