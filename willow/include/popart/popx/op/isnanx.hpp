#ifndef GUARD_NEURALNET_ISNANX_HPP
#define GUARD_NEURALNET_ISNANX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class IsNaNx : public ElementWiseUnaryOpx {
public:
  IsNaNx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
