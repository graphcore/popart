#ifndef GUARD_NEURALNET_NEGATEX_HPP
#define GUARD_NEURALNET_NEGATEX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

namespace popx {

class NegateOpx : public ElementWiseUnaryOpx {
public:
  NegateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class NegateGradOpx : public ElementWiseUnaryOpx {
public:
  NegateGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
