#ifndef GUARD_NEURALNET_STASHX_HPP
#define GUARD_NEURALNET_STASHX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class StashOp;

namespace popx {

class StashOpx : public Opx {
public:
  StashOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
