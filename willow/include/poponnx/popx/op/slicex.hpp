#ifndef GUARD_NEURALNET_SLICEX_HPP
#define GUARD_NEURALNET_SLICEX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/padx.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class SliceOp;

namespace popx {

class SliceOpx : public Opx {
public:
  SliceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  SliceOp *getSliceOp() const;
};

class SliceGradOpx : public PadOpx {
public:
  SliceGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace poponnx

#endif
