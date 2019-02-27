#ifndef GUARD_NEURALNET_GROUPNORMX_HPP
#define GUARD_NEURALNET_GROUPNORMX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/normx.hpp>

namespace poponnx {

namespace popx {

class GroupNormOpx : public NormOpx {
public:
  GroupNormOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
};

class GroupNormGradOpx : public NormOpx {
public:
  GroupNormGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
};

} // namespace popx
} // namespace poponnx

#endif
