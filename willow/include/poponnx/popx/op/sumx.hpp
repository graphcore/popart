#ifndef GUARD_NEURALNET_SUMX_HPP
#define GUARD_NEURALNET_SUMX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class SumOp;

namespace popx {

class SumOpx : public Opx {
public:
  SumOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
};

} // namespace popx
} // namespace poponnx

#endif
