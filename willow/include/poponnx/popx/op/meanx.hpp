#ifndef GUARD_NEURALNET_MEANX_HPP
#define GUARD_NEURALNET_MEANX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

namespace popx {

class MeanOpx : public ElementWiseUnaryOpx {
public:
  MeanOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class MeanArgGradOpx : public Opx {
public:
  MeanArgGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
