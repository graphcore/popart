// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CTCBEAMSEARCHX_HPP
#define GUARD_NEURALNET_CTCBEAMSEARCHX_HPP

#include <memory>

#include <popart/popx/opxmanager.hpp>

namespace popnn {
namespace ctc {
class Plan;
} // namespace ctc
} // namespace popnn

namespace popart {
namespace popx {
class CtcBeamSearchDecoderOpx : public PopOpx {
public:
  CtcBeamSearchDecoderOpx(Op *op, Devicex *device);

  void grow(snap::program::Sequence &prog) const final;

private:
  // Unique pointer so we can forward-declare to avoid including poplar headers.
  std::unique_ptr<popnn::ctc::Plan> plan;
};
} // namespace popx
} // namespace popart

#endif // GUARD_NEURALNET_CTCBEAMSEARCHX_HPP