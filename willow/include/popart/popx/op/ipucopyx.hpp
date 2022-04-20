// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_IPUCOPYX_HPP
#define GUARD_NEURALNET_IPUCOPYX_HPP

#include <snap/Graph.hpp>
#include <snap/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/namesx.hpp>
#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class IpuCopyOpx : public PopOpx {
public:
  IpuCopyOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  // When pipelining is enabled, `IpuCopyOpx::grow` is not used.
  // `createPipelinedOutput` is used in place of grow, and created the
  // destination tensor for the copy.
  PreparedCopyTensors createPipelinedOutput() const;
  // `growPipelined` add the copy program to the input Sequence. This is called
  // for every pipeline cycle the copy appears in.
  void growPipelined(snap::program::Sequence &, PreparedCopyTensors) const;

  InputCreatorType getInputCreatorType(InIndex index) const final {
    return InputCreatorType::CanUnwind;
  }
  bool canUnwind(InIndex in, OutIndex out) const final { return in == out; }
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;

  snap::Graph &srcVirtualGraph(InIndex) const final;
  snap::Graph &dstVirtualGraph(OutIndex) const final;
};

} // namespace popx
} // namespace popart

#endif
