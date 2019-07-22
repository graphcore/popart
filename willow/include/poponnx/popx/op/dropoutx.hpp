#ifndef GUARD_NEURALNET_DROPOUTX_HPP
#define GUARD_NEURALNET_DROPOUTX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

class DropoutOp;

namespace popx {

class DropoutOpx : public ElementWiseUnaryOpx {
public:
  DropoutOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
