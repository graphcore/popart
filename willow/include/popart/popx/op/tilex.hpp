#ifndef GUARD_NEURALNET_TILEX_HPP
#define GUARD_NEURALNET_TILEX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class TileOp;
class TileGradOp;

namespace popx {

class TileOpx : public Opx {
public:
  TileOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // Design decision: InputCreatorType could be CANUNWIND, but not overriding
  // default, DEADEND. The unwind function would slice the output over each
  // dimension with a non-idendity repeat value. This could result in allocating
  // a much larger tensor than required by the input's shape
};

class TileGradOpx : public Opx {
public:
  TileGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
