// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

Opx::Opx(Op *op_p_, Devicex *dv_p_) : PopOpx(op_p_, dv_p_) {}

Opx::~Opx() = default;

poplar::Tensor Opx::createInput(InIndex index,
                                const poplar::DebugNameAndId &dnai) const {
  return PopOpx::createInputTensor(index, dnai).getPoplarTensor();
}

snap::Tensor Opx::createInputTensor(InIndex index,
                                    const poplar::DebugNameAndId &dnai) const {
  return snap::Tensor{createInput(index, dnai), PopOpx::graph()};
}

poplar::Tensor
Opx::unwindTensorLayout(poplar::Tensor tensor, InIndex in, OutIndex out) const {
  return PopOpx::unwindTensorLayout(
             snap::Tensor{tensor, PopOpx::graph()}, in, out)
      .getPoplarTensor();
}

snap::Tensor
Opx::unwindTensorLayout(snap::Tensor tensor, InIndex in, OutIndex out) const {
  return snap::Tensor{unwindTensorLayout(tensor.getPoplarTensor(), in, out),
                      PopOpx::graph()};
}

bool Opx::createsEquiv(int, const Opx *, int) const {
  throw error("No check for equivalent tensor create for type {}", op_p->opid);
}

poplar::Graph &Opx::graph() const { return PopOpx::graph().getPoplarGraph(); }

const poplar::Tensor &Opx::get(TensorId id) const {
  return PopOpx::get(id).getPoplarTensor();
}

const poplar::Tensor &Opx::getView(TensorId id) const {
  return PopOpx::getView(id).getPoplarTensor();
}

void Opx::insert(TensorId id, const poplar::Tensor &tensor) const {
  PopOpx::insert(id, snap::Tensor{tensor, dv_p->lowering().graph()});
}

const poplar::Tensor &Opx::getInTensor(InIndex index) const {
  return PopOpx::getInTensor(index).getPoplarTensor();
}

const poplar::Tensor &Opx::getOutTensor(OutIndex index) const {
  return PopOpx::getOutTensor(index).getPoplarTensor();
}

const poplar::Tensor &Opx::getInView(InIndex index) const {
  return PopOpx::getInView(index).getPoplarTensor();
}

const poplar::Tensor &Opx::getOutView(OutIndex index) const {
  return PopOpx::getOutView(index).getPoplarTensor();
}

void Opx::setOutTensor(OutIndex index, const poplar::Tensor &t) const {
  PopOpx::setOutTensor(index, snap::Tensor{t, dv_p->lowering().graph()});
}

} // namespace popx
} // namespace popart
