#ifndef GUARD_NEURALNET_SIGNX_HPP
#define GUARD_NEURALNET_SIGNX_HPP

#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class SignOpx : public ElementWiseUnaryOpx {
public:
  SignOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class SignGradOpx : public Opx {
public:
  SignGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
