#ifndef GUARD_NEURALNET_REDUCEL1X_HPP
#define GUARD_NEURALNET_REDUCEL1X_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class ReduceL1Op;

namespace popx {

class ReduceL1Opx : public Opx {
public:
  ReduceL1Opx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceL1GradOpx : public Opx {
public:
  ReduceL1GradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
