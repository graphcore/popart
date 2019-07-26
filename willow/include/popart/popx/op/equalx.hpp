#ifndef GUARD_NEURALNET_EQUALX_HPP
#define GUARD_NEURALNET_EQUALX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

class EqualOp;

namespace popx {

class EqualOpx : public BinaryComparisonOpx {
public:
  EqualOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
