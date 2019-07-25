#ifndef GUARD_NEURALNET_GROUPNORMX_HPP
#define GUARD_NEURALNET_GROUPNORMX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/normx.hpp>

namespace popart {

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
} // namespace popart

#endif
