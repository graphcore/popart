#ifndef GUARD_NEURALNET_IDENTITYX_HPP
#define GUARD_NEURALNET_IDENTITYX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class IdentityOpx : public ElementWiseUnaryOpx {
public:
  IdentityOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class IdentityInplaceOpx : public Opx {
public:
  IdentityInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class IdentityGradOpx : public ElementWiseUnaryOpx {
public:
  IdentityGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
