#ifndef GUARD_NEURALNET_GETRANDOMSEEDX_HPP
#define GUARD_NEURALNET_GETRANDOMSEEDX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class GetRandomSeedOp;

namespace popx {

class GetRandomSeedOpx : public Opx {
public:
  GetRandomSeedOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
