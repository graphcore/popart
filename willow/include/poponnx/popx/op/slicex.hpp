#ifndef GUARD_NEURALNET_SLICEX_HPP
#define GUARD_NEURALNET_SLICEX_HPP

#include <poponnx/popx/opx.hpp>

namespace poponnx {

class SliceOp;
class SliceInplaceOp;

namespace popx {

class SliceOpx : public Opx {
public:
  SliceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  SliceOp *getSliceOp() const;
};

class SliceInplaceOpx : public Opx {
public:
  SliceInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  SliceInplaceOp *getSliceInplaceOp() const;
};

} // namespace popx
} // namespace poponnx

#endif
