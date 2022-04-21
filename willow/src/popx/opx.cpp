// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/timepartitionlogger.hpp>
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

poplar::Tensor Opx::cloneNcopy(poplar::program::Sequence &prog,
                               TensorId id) const {
  const poplar::Tensor &tensor = get(id);
  return cloneNcopy(prog, tensor, id + "[cloned]");
}

poplar::Tensor Opx::cloneNcopy(poplar::program::Sequence &prog,
                               const poplar::Tensor &tensor,
                               std::string name) const {

  const auto scopedTimer =
      getDevicex()->ir().timePartitionLogger().scopedStopwatch(
          "Clone (and copy)");

  // TODO Would be good to get the name of the tensor
  auto outTensor = graph().clone(tensor, debugContext(name));
  prog.add(poplar::program::Copy(tensor, outTensor, false, debugContext()));
  return outTensor;
}

poplar::Tensor Opx::broadcast(const std::vector<int64_t> &desired_shape,
                              TensorId id) const {
  return broadcast(desired_shape, get(id));
}

poplar::Tensor Opx::broadcast(const std::vector<int64_t> &desired_shape,
                              poplar::Tensor t) const {
  const auto &t_shape = t.shape();

  // `new_shape` is `t_shape` prepended with ones to have matching rank as
  // `desired_shape`
  std::vector<std::size_t> new_shape(desired_shape.size(), 1);
  std::copy(t_shape.rbegin(), t_shape.rend(), new_shape.rbegin());

  // `t` now has matching rank as `desired_shape`
  t = t.reshape(new_shape);

  // Iteratively broadcast each mismatched dimension of `t`. This will
  // result in the shape of `t` matching `desired_shape`.
  for (int dim = 0; dim < desired_shape.size(); ++dim) {
    if (new_shape[dim] != desired_shape[dim] && new_shape[dim] != 1) {
      // Incompatible dimension found. Throw an exception, borrowing the same
      // terminology as numpy.
      throw error("np broadcasting failed, frames are not aligned");
    }

    if (new_shape[dim] != desired_shape[dim]) {
      t = t.broadcast(static_cast<unsigned>(desired_shape[dim]), dim);
    }
  }

  return t;
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

poplar::Tensor Opx::getConst(const poplar::Type &type,
                             const std::vector<size_t> &shape,
                             double val,
                             const std::string &name) const {
  return PopOpx::getConst(type, shape, val, name).getPoplarTensor();
}

poplar::Tensor Opx::getScalarVariable(const poplar::Type &type,
                                      const std::string &name) const {
  return PopOpx::getScalarVariable(type, name).getPoplarTensor();
}

void Opx::grow(snap::program::Sequence &prog) const {
  return grow(prog.getPoplarSequence());
}

void Opx::grow(poplar::program::Sequence &) const {
  throw error("adding poplar::Tensors not implemented for {}", op_p->opid);
}

} // namespace popx
} // namespace popart
