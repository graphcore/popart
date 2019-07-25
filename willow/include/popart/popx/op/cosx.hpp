#ifndef GUARD_NEURALNET_COSX_HPP
#define GUARD_NEURALNET_COSX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class CosOpx : public ElementWiseUnaryOpx {
public:
  CosOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
