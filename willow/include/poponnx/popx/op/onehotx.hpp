#ifndef GUARD_NEURALNET_ONEHOTX_HPP
#define GUARD_NEURALNET_ONEHOTX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class OnehotOpx : public Opx {
public:
  OnehotOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class OnehotGradOpx : public Opx {
public:
  OnehotGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
