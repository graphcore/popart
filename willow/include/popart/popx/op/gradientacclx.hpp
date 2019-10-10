#ifndef GUARD_NEURALNET_GRADIENTACCLX_HPP
#define GUARD_NEURALNET_GRADIENTACCLX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class GradientAcclOpx : public Opx {
public:
  GradientAcclOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  poplar::Tensor createInput(int index, const std::string &name) const override;
  InputCreatorType getInputCreatorType(int index) const override;
  bool createsEquiv(int, const Opx *, int) const final;
  std::vector<TensorId> mustExistBeforeCreate(int index) const override;
};

class ResetAcclOpx : public Opx {
public:
  ResetAcclOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
