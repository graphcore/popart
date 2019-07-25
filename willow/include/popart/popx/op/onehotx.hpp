#ifndef GUARD_NEURALNET_ONEHOTX_HPP
#define GUARD_NEURALNET_ONEHOTX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

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
} // namespace popart

#endif
