#ifndef GUARD_NEURALNET_SGD1ACCLREDUCEX_HPP
#define GUARD_NEURALNET_SGD1ACCLREDUCEX_HPP

#include <popart/op.hpp>
#include <popart/popx/op/varupdatex.hpp>

namespace popart {
namespace popx {

class SGD1AcclReduceOpx : public VarUpdateOpx {
public:
  SGD1AcclReduceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // does not create inputs
};

} // namespace popx
} // namespace popart

#endif
