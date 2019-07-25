#ifndef GUARD_NEURALNET_NORMX_HPP
#define GUARD_NEURALNET_NORMX_HPP

#include <popart/names.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/normx.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <poplar/Tensor.hpp>

// Base class for the norm options such as  group, instance, batch

namespace poplar {
using Shape = std::vector<std::size_t>;
}

namespace popart {
namespace popx {

class NormOpx : public Opx {
public:
  NormOpx(Op *, Devicex *);

protected:
  std::pair<poplar::Tensor, poplar::Shape>
  convertOnnxInputToPoplarInput(const poplar::Tensor &onnxInput) const;

  poplar::Tensor convertPoplarOutputToOnnxOutput(
      const poplar::Tensor &poplarOutput,
      const poplar::Shape &nonBroadcastDimensions) const;

  poplar::Tensor convertInvSdToVar(poplar::program::Sequence &prog,
                                   const poplar::Tensor &invSd,
                                   float epsilon) const;

  poplar::Tensor convertVarToInvSd(poplar::program::Sequence &prog,
                                   const poplar::Tensor &var,
                                   float epsilon) const;

private:
};

} // namespace popx
} // namespace popart

#endif
