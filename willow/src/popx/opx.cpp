// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/conv.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/viewchangers.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {
namespace popx {

Opx::Opx(Op *op_p_, Devicex *dv_p_) : op_p(op_p_), dv_p(dv_p_) {}

Opx::~Opx() = default;

poplar::Tensor Opx::createInput(InIndex index, const std::string &name) const {
  throw error("Opx for {} cannot create Input index:{} name:{}",
              op_p->opid,
              index,
              name);
}

bool Opx::createsEquiv(int, const Opx *, int) const {
  throw error("No check for equivalent tensor create for type {}", op_p->opid);
}

std::vector<TensorId> Opx::mustExistBeforeCreate(int index0) const {
  throw error("Opx for {} cannot say which poplar Tensors must exist to create "
              "at index {}",
              op_p->opid,
              index0);
}

void Opx::grow(poplar::program::Sequence &) const {
  throw error("adding poplar::Tensors not implemented for {}", op_p->opid);
}

InputCreatorType Opx::getInputCreatorType(InIndex) const {
  return InputCreatorType::Deadend;
}

bool Opx::canUnwind(InIndex in, OutIndex) const {
  auto type = getInputCreatorType(in);
  return type == InputCreatorType::CanUnwind ||
         type == InputCreatorType::CanCreateOrUnwind;
}

poplar::Tensor
Opx::unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const {
  throw error("Opx for {} cannot unwind the tensor layout change between input "
              "and output for {}",
              op_p->opid);
}

view::RegMap Opx::unwindRegion(InIndex, OutIndex) const {
  throw error("Opx cannot unwind the region between input "
              "and output for {}",
              op_p->opid);
}

bool Opx::hasCreatorViewChangers(InIndex) const { return false; }

ViewChangers Opx::getCreatorViewChangers(InIndex) const {
  return ViewChangers();
}

bool Opx::outputCreatedExternally(OutIndex index) const { return false; }

int64_t Opx::getVirtualGraphId() const {
  if (op_p->hasVirtualGraphId()) {
    return op_p->getVirtualGraphId();
  } else {
    if (op_p->getIr().virtualGraphsEnabled()) {
      throw error("{} does not have a virtual graph attribute",
                  op_p->debugName());
    } else {
      return 0;
    }
  }
}

poplar::Graph &Opx::graph() const {
  if (op_p->getIr().virtualGraphsEnabled()) {
    return dv_p->getVirtualGraph(getVirtualGraphId(),
                                 op_p->settings.useIoTiles);
  } else {
    return dv_p->graph();
  }
}

poplar::Graph &Opx::srcGraph(InIndex) const { return graph(); }

poplar::Graph &Opx::dstGraph(OutIndex) const { return graph(); }

const poplar::Tensor &Opx::get(TensorId id) const {
  return dv_p->tensors.get(id);
}

const poplar::Tensor &Opx::getView(TensorId id) const {
  return dv_p->tensors.getView(id);
}

void Opx::insert(TensorId id, const poplar::Tensor &tensor) const {
  dv_p->tensors.insert(id, tensor);
}

TensorId Opx::inId(InIndex index) const { return op_p->input->id(index); }
TensorId Opx::outId(OutIndex index) const { return op_p->output->id(index); }

bool Opx::hasInput(InIndex index) const { return op_p->input->hasIndex(index); }

bool Opx::hasOutput(OutIndex index) const {
  return op_p->output->hasIndex(index);
}

const poplar::Tensor &Opx::getInTensor(InIndex index) const {
  if (!cachedInputs.empty()) {
    return cachedInputs[index];
  } else {
    return get(op_p->input->id(index));
  }
}

const poplar::Tensor &Opx::getOutTensor(OutIndex index) const {
  if (cachedOutputs && !cachedOutputs->empty()) {
    return (*cachedOutputs)[index];
  } else {
    return get(op_p->output->id(index));
  }
}

const poplar::Tensor &Opx::getInView(InIndex index) const {
  return getView(op_p->input->id(index));
}

const poplar::Tensor &Opx::getOutView(OutIndex index) const {
  return getView(op_p->output->id(index));
}

bool Opx::hasInViewChangers(InIndex index) const {
  return dv_p->tensors.hasViewChangers(op_p->input->id(index));
}

const ViewChangers &Opx::getInViewChangers(InIndex index) const {
  return dv_p->tensors.getViewChangers(op_p->input->id(index));
}

void Opx::setOutViewChangers(OutIndex index,
                             const ViewChangers &changers) const {
  return dv_p->tensors.setViewChangers(op_p->output->id(index), changers);
}

void Opx::setOutTensor(OutIndex index, const poplar::Tensor &tensor) const {
  // Assume that if we have cached inputs then we will use cached outputs
  if (cachedOutputs) {
    cachedOutputs->insert(cachedOutputs->begin() + index, tensor);
  } else {
    logging::trace("Op {} inserting poplar::Tensor {}",
                   getOp<Op>().debugName(),
                   op_p->output->id(index));
    insert(op_p->output->id(index), tensor);
  }
}

Tensor *Opx::inTensor(InIndex index) const {
  return op_p->input->tensor(index);
}
Tensor *Opx::outTensor(OutIndex index) const {
  return op_p->output->tensor(index);
}

const TensorInfo &Opx::inInfo(InIndex index) const {
  return inTensor(index)->info;
}

const Shape &Opx::inShape(InIndex index) const { return inInfo(index).shape(); }

const TensorInfo &Opx::outInfo(OutIndex index) const {
  return outTensor(index)->info;
}

const Shape &Opx::outShape(OutIndex index) const {
  return outInfo(index).shape();
}

// If the operator has been named return the name, (i.e. "my_add/23")
// else return the id (i.e "23")
std::string Opx::idStr() const {
  if (!op_p->name().empty()) {
    return op_p->name() + sNameDelimiter + std::to_string(op_p->id);
  } else {
    return std::to_string(op_p->id);
  }
}

std::string Opx::debugPrefix(const std::string &prefix) const {
  return idStr() + sNameDelimiter + prefix;
}

std::string Opx::debugPrefix(const std::string &p1,
                             const std::string &p2) const {
  return debugPrefix(p1 + sNameDelimiter + p2);
}

std::string Opx::debugPrefix() const { return idStr(); }

poplar::Tensor Opx::cloneNcopy(poplar::program::Sequence &prog,
                               TensorId id) const {
  auto outTensor = graph().clone(get(id));
  poplar::program::Copy copyProg(get(id), outTensor);
  prog.add(copyProg);
  return outTensor;
}

poplar::Tensor Opx::cloneNcopy(poplar::program::Sequence &prog,
                               const poplar::Tensor &tensor) const {
  auto outTensor = graph().clone(tensor);
  prog.add(poplar::program::Copy(tensor, outTensor));
  return outTensor;
}

poplar::Tensor Opx::broadcast(const std::vector<int64_t> &desired_shape,
                              TensorId id) const {
  return broadcast(desired_shape, get(id));
}

const Devicex *Opx::getDevicex() const { return dv_p; }

std::map<uint32_t, poplar::Tensor> &Opx::getDropoutReferenceTensors() const {
  return dv_p->dropoutReferenceTensors;
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

poplar::Tensor Opx::getConst(const poplar::Type &type,
                             const std::vector<size_t> &shape,
                             double val,
                             const std::string &name) const {
  return dv_p->getConst(graph(), type, shape, val, name);
}

poplar::Tensor Opx::getScalarVariable(const poplar::Type &type,
                                      const std::string &name) const {
  return dv_p->getScalarVariable(graph(), type, name);
}

std::vector<std::tuple<TensorId, TensorId, bool>>
Opx::getOutputsToPrepare() const {
  return {};
}

} // namespace popx
} // namespace popart
