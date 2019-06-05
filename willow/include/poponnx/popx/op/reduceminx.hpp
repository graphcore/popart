#ifndef GUARD_NEURALNET_REDUCEMINX_HPP
#define GUARD_NEURALNET_REDUCEMINX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class ReduceMinOp;

namespace popx {

class ReduceMinOpx : public Opx {
public:
  ReduceMinOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceMinGradOpx : public Opx {
public:
  ReduceMinGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace poponnx

#endif
