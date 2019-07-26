#ifndef GUARD_NEURALNET_ANDX_HPP
#define GUARD_NEURALNET_ANDX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

class AndOp;

namespace popx {

class AndOpx : public BinaryComparisonOpx {
public:
  AndOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
