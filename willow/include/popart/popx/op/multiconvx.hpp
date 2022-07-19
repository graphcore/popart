// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_MULTICONVX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_MULTICONVX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <snap/Tensor.hpp>
#include <vector>
#include <poplar/OptionFlags.hpp>
#include <popart/popx/op/convbasex.hpp>

namespace poplin {
namespace multiconv {
struct CreateTensorArgs;
} // namespace multiconv
} // namespace poplin
namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class MultiConvOpx : public MultiConvBaseOpx {
public:
  MultiConvOpx(Op *, Devicex *);

  snap::Tensor createWeightsInput(const poplar::DebugNameAndId &dnai,
                                  int convIndex) const final;
  snap::Tensor createDataInput(const poplar::DebugNameAndId &dnai,
                               int convIndex) const final;
  std::vector<snap::Tensor>
  convolve(snap::program::Sequence &,
           const std::vector<snap::Tensor> &) const final;

private:
  std::vector<poplin::multiconv::CreateTensorArgs>
  getCreateTensorArgs(const poplar::DebugNameAndId &dnai) const;
  poplar::OptionFlags getGlobalOptions() const;
};

class MultiConvWeightsGradOpx : public MultiConvWeightsGradBaseOpx {
public:
  MultiConvWeightsGradOpx(Op *, Devicex *);
  std::vector<snap::Tensor>
  calculateWeightDeltas(snap::program::Sequence &) const final;

private:
  poplar::OptionFlags getGlobalOptions() const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_MULTICONVX_HPP_
