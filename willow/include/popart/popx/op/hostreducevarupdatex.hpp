#ifndef GUARD_NEURALNET_HOSTREDUCEVARUPDATEX_HPP
#define GUARD_NEURALNET_HOSTREDUCEVARUPDATEX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/varupdatex.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {
class GradCopyToHostOpx : public Opx {
public:
  GradCopyToHostOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class GradCopyFromHostOpx : public Opx {
public:
  GradCopyFromHostOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class HostReduceVarCopyOpx : public VarUpdateOpx {
public:
  HostReduceVarCopyOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
