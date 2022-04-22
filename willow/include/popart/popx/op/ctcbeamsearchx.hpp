// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CTCBEAMSEARCHX_HPP
#define GUARD_NEURALNET_CTCBEAMSEARCHX_HPP

#include <memory>

#include "popart/popx/popopx.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popnn {
namespace ctc {
class Plan;
} // namespace ctc
} // namespace popnn

namespace popart {
class Op;

namespace popx {
class Devicex;

class CtcBeamSearchDecoderOpx : public PopOpx {
public:
  CtcBeamSearchDecoderOpx(Op *op, Devicex *device);
  ~CtcBeamSearchDecoderOpx();

  void grow(snap::program::Sequence &prog) const final;

private:
  // Unique pointer so we can forward-declare to avoid including poplar headers.
  std::unique_ptr<popnn::ctc::Plan> plan;
};
} // namespace popx
} // namespace popart

#endif // GUARD_NEURALNET_CTCBEAMSEARCHX_HPP
