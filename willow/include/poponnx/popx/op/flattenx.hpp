#ifndef GUARD_NEURALNET_FLATTENX_HPP
#define GUARD_NEURALNET_FLATTENX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/reshapex.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class FlattenAliasOpx : public Opx {
public:
  FlattenAliasOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;

protected:
  poplar::Tensor grow(poplar::program::Sequence &, poplar::Tensor &) const;
};

class FlattenOpx : public FlattenAliasOpx {
public:
  using FlattenAliasOpx::FlattenAliasOpx;
  void grow(poplar::program::Sequence &) const final;
};

class FlattenGradOpx : public ReshapeOpx {
public:
  FlattenGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace poponnx

#endif
