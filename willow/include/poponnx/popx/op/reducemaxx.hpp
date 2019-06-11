#ifndef GUARD_NEURALNET_REDUCEMAXX_HPP
#define GUARD_NEURALNET_REDUCEMAXX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class ReduceMaxOp;

namespace popx {

class ReduceMaxOpx : public Opx {
public:
  ReduceMaxOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceMaxGradOpx : public Opx {
public:
  ReduceMaxGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace poponnx

#endif
