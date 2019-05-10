#ifndef GUARD_NEURALNET_FLATTENX_HPP
#define GUARD_NEURALNET_FLATTENX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/reshapex.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class FlattenInplaceOpx : public Opx {
public:
  FlattenInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;

private:
  std::vector<size_t> outShape;
};

class FlattenOpx : public Opx {
public:
  FlattenOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  std::vector<size_t> outShape;
};

class FlattenGradOpx : public ReshapeOpx {
public:
  FlattenGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace poponnx

#endif
