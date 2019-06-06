#ifndef GUARD_NEURALNET_REDUCEL2X_HPP
#define GUARD_NEURALNET_REDUCEL2X_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class ReduceL2Op;

namespace popx {

class ReduceL2Opx : public Opx {
public:
  ReduceL2Opx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceL2GradOpx : public Opx {
public:
  ReduceL2GradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace poponnx

#endif
