#ifndef GUARD_NEURALNET_ISINFX_HPP
#define GUARD_NEURALNET_ISINFX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class IsInfx : public ElementWiseUnaryOpx {
public:
  IsInfx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
