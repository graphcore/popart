#ifndef GUARD_NEURALNET_SLICEX_HPP
#define GUARD_NEURALNET_SLICEX_HPP

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

} // namespace popx
} // namespace poponnx

#endif
