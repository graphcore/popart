// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

Opx::Opx(Op *op_p_, Devicex *dv_p_) : PopOpx(op_p_, dv_p_) {}

Opx::~Opx() = default;

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
