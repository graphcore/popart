#ifndef GUARD_NEURALNET_DROPOUTX_HPP
#define GUARD_NEURALNET_DROPOUTX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

class DropoutOp;

namespace popx {

class DropoutOpx : public ElementWiseUnaryOpx {
public:
  DropoutOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
