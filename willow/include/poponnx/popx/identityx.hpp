#ifndef GUARD_NEURALNET_IDENTITYX_HPP
#define GUARD_NEURALNET_IDENTITYX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace willow {

namespace popx {

class IdentityOpx : public Opx {
public:
  IdentityOpx(Op *, Devicex *);
  virtual void grow(poplar::program::Sequence &) const override final;
};

class IdentityGradOpx : public IdentityOpx {
public:
  IdentityGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace willow

#endif
