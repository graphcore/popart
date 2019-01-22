#ifndef GUARD_NEURALNET_SIGMOIDX_HPP
#define GUARD_NEURALNET_SIGMOIDX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

namespace popx {

class SigmoidOpx : public ElementWiseUnaryOpx {
public:
  SigmoidOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class SigmoidGradOpx : public ElementWiseUnaryOpx {
public:
  SigmoidGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
