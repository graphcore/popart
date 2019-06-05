#ifndef GUARD_NEURALNET_SplitX_HPP
#define GUARD_NEURALNET_SplitX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class SplitOpx : public Opx {
public:
  SplitOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
