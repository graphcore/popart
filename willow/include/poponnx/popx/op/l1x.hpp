#ifndef GUARD_NEURALNET_L1X_HPP
#define GUARD_NEURALNET_L1X_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class L1GradOp;
class L1Op;

namespace popx {

class L1Opx : public Opx {
public:
  L1Opx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
};

class L1GradOpx : public Opx {
public:
  L1GradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
