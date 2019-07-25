#ifndef GUARD_NEURALNET_RESTOREX_HPP
#define GUARD_NEURALNET_RESTOREX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class RestoreOp;

namespace popx {

class RestoreOpx : public Opx {
public:
  RestoreOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class RestoreInplaceOpx : public Opx {
public:
  RestoreInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
