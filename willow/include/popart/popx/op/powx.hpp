#ifndef GUARD_NEURALNET_POWX_HPP
#define GUARD_NEURALNET_POWX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/op/reducesumx.hpp>

namespace popart {

class PowOp;

namespace popx {

class PowOpx : public ElementWiseBinaryOpx {
public:
  PowOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
