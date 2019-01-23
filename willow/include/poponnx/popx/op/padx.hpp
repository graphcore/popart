#ifndef GUARD_NEURALNET_PADX_HPP
#define GUARD_NEURALNET_PADX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class PadOp;

namespace popx {

class PadOpx : public Opx {
public:
  PadOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;

private:
  PadOp *getPadOp() const;
};

} // namespace popx
} // namespace poponnx

#endif
