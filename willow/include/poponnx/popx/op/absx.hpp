#ifndef GUARD_NEURALNET_ABSX_HPP
#define GUARD_NEURALNET_ABSX_HPP

#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {
namespace popx {

class AbsOpx : public ElementWiseUnaryOpx {
public:
  AbsOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
