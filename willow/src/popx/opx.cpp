// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

Opx::Opx(Op *op_p_, Devicex *dv_p_) : PopOpx(op_p_, dv_p_) {}

Opx::~Opx() = default;

poplar::Tensor Opx::createInput(InIndex index,
                                const poplar::DebugNameAndId &dnai) const {
  return createInputTensor(index, dnai).getPoplarTensor();
}

poplar::Tensor
Opx::unwindTensorLayout(poplar::Tensor tensor, InIndex in, OutIndex out) const {
  return PopOpx::unwindTensorLayout(
             snap::Tensor{tensor, PopOpx::graph()}, in, out)
      .getPoplarTensor();
}

bool Opx::createsEquiv(int, const Opx *, int) const {
  throw error("No check for equivalent tensor create for type {}", op_p->opid);
}

poplar::Graph &Opx::graph() const { return PopOpx::graph().getPoplarGraph(); }

poplar::Graph &Opx::srcGraph(InIndex index) const {
  return srcVirtualGraph(index).getPoplarGraph();
}

poplar::Graph &Opx::dstGraph(OutIndex index) const {
  return dstVirtualGraph(index).getPoplarGraph();
}

} // namespace popx
} // namespace popart
