#ifndef GUARD_NEURALNET_IDENTITYX_HPP
#define GUARD_NEURALNET_IDENTITYX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

namespace popx {

class IdentityOpx : public ElementWiseUnaryOpx {
public:
  IdentityOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class IdentityGradOpx : public ElementWiseUnaryOpx {
public:
  IdentityGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
