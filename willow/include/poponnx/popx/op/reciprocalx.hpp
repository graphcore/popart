#ifndef GUARD_NEURALNET_RECIPROCALX_HPP
#define GUARD_NEURALNET_RECIPROCALX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

namespace popx {

class ReciprocalOpx : public ElementWiseUnaryOpx {
public:
  ReciprocalOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
