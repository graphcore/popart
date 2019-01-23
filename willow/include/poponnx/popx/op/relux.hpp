#ifndef GUARD_NEURALNET_RELUX_HPP
#define GUARD_NEURALNET_RELUX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class ReluOp;
class ReluInplaceOp;
class ReluGradOp;

namespace popx {

class ReluOpx : public Opx {
public:
  ReluOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
};

class ReluInplaceOpx : public Opx {
public:
  ReluInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
};

class ReluGradOpx : public Opx {
public:
  ReluGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
