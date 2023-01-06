// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_IPUCOPYX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_IPUCOPYX_HPP_

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/namesx.hpp>
#include <popart/popx/opx.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class IpuCopyOpx : public Opx {
public:
  IpuCopyOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // When pipelining is enabled, `IpuCopyOpx::grow` is not used.
  // `createPipelinedOutput` is used in place of grow, and created the
  // destination tensor for the copy.
  PreparedCopyTensors createPipelinedOutput() const;
  // `growPipelined` add the copy program to the input Sequence. This is called
  // for every pipeline cycle the copy appears in.
  void growPipelined(poplar::program::Sequence &, PreparedCopyTensors) const;

  InputCreatorType getInputCreatorType(InIndex index) const final {
    return InputCreatorType::CanUnwind;
  }
  bool canUnwind(InIndex in, OutIndex out) const final { return in == out; }
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;

  poplar::Graph &srcGraph(InIndex) const final;
  poplar::Graph &dstGraph(OutIndex) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_IPUCOPYX_HPP_
