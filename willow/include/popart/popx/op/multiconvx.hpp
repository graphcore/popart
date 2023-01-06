// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_MULTICONVX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_MULTICONVX_HPP_

#include <vector>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>
#include <popart/popx/op/convbasex.hpp>

#include "popart/popx/debugcontextx.hpp"

namespace poplin {
namespace multiconv {
struct CreateTensorArgs;
} // namespace multiconv
} // namespace poplin
namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class MultiConvOpx : public MultiConvBaseOpx {
public:
  MultiConvOpx(Op *, Devicex *);

  poplar::Tensor createWeightsInput(const poplar::DebugNameAndId &dnai,
                                    int convIndex) const final;
  poplar::Tensor createDataInput(const poplar::DebugNameAndId &dnai,
                                 int convIndex) const final;
  std::vector<poplar::Tensor>
  convolve(poplar::program::Sequence &,
           const std::vector<poplar::Tensor> &) const final;

private:
  std::vector<poplin::multiconv::CreateTensorArgs>
  getCreateTensorArgs(const poplar::DebugNameAndId &dnai) const;
  poplar::OptionFlags getGlobalOptions() const;
};

class MultiConvWeightsGradOpx : public MultiConvWeightsGradBaseOpx {
public:
  MultiConvWeightsGradOpx(Op *, Devicex *);
  std::vector<poplar::Tensor>
  calculateWeightDeltas(poplar::program::Sequence &) const final;

private:
  poplar::OptionFlags getGlobalOptions() const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_MULTICONVX_HPP_
