#ifndef GUARD_NEURALNET_SUBSAMPLEX_HPP
#define GUARD_NEURALNET_SUBSAMPLEX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class SubsampleOpx : public Opx {

public:
  SubsampleOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class SubsampleInplaceOpx : public Opx {

public:
  SubsampleInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class SubsampleGradOpx : public Opx {
public:
  SubsampleGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
