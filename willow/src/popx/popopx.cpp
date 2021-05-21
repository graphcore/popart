// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/timepartitionlogger.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/conv.hpp>
#include <popart/popx/debugcontextx.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/viewchangers.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {
namespace popx {

PopOpx::PopOpx(Op *op_p_, Devicex *dv_p_) : op_p(op_p_), dv_p(dv_p_) {}

PopOpx::~PopOpx() = default;

poplar::Tensor PopOpx::createInput(InIndex index,
                                   const poplar::DebugNameAndId &dnai) const {
  throw error("PopOpx for {} cannot create Input index:{} name:{}",
              op_p->opid,
              index,
              dnai.getPathName());
}

std::set<TensorId> PopOpx::mustExistBeforeCreate(int index0) const {
  throw error(
      "PopOpx for {} cannot say which poplar Tensors must exist to create "
      "at index {}",
      op_p->opid,
      index0);
}

DnfTensorIds PopOpx::mustExistBeforeCreateDNF(int index0) const {
  return {mustExistBeforeCreate(index0)};
}

void PopOpx::grow(poplar::program::Sequence &) const {
  throw error("adding poplar::Tensors not implemented for {}", op_p->opid);
}

void PopOpx::grow(std::vector<poplar::program::Sequence> &sequences) const {
  if (sequences.empty()) {
    auto partitioner  = dv_p->lowering().getSubgraphPartitioner();
    auto subgraphPart = partitioner->getOpSubgraphPartBegin(op_p);

    std::stringstream ss;
    ss << op_p->getGraph().id.str() << "/" << subgraphPart;
    sequences.resize(1, poplar::program::Sequence({}, debugContext(ss.str())));
  }

  // By default, use the PopOpx::grow(poplar::program::Sequence &) function.
  // Currently, only CallOpx overloads this PopOpx::grow method to grow over
  // multiple fragments.
  grow(*sequences.begin());
}

InputCreatorType PopOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::Deadend;
}

bool PopOpx::canUnwind(InIndex in, OutIndex) const {
  auto type = getInputCreatorType(in);
  return type == InputCreatorType::CanUnwind ||
         type == InputCreatorType::CanCreateOrUnwind;
}

poplar::Tensor
PopOpx::unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const {
  throw error(
      "PopOpx for {} cannot unwind the tensor layout change between input "
      "and output for {}",
      op_p->opid);
}

view::RegMap PopOpx::unwindRegion(InIndex, OutIndex) const {
  throw error("PopOpx cannot unwind the region between input "
              "and output for {}",
              op_p->opid);
}

bool PopOpx::hasCreatorViewChangers(InIndex) const { return false; }

ViewChangers PopOpx::getCreatorViewChangers(InIndex) const {
  return ViewChangers();
}

bool PopOpx::outputCreatedExternally(OutIndex index) const { return false; }

int64_t PopOpx::getVirtualGraphId() const {
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

snap::Graph &PopOpx::graph() const {
  if (op_p->getIr().virtualGraphsEnabled()) {
    return dv_p->lowering().getVirtualGraph(getVirtualGraphId(),
                                            op_p->settings.tileSet);
  } else {
    return dv_p->lowering().graph();
  }
}

snap::Graph &PopOpx::srcVirtualGraph(InIndex index) const {
  auto &op  = getOp<Op>();
  auto vgid = op.getIntrospectionInVirtualGraphId(index);
  if (vgid.first == unusedVGraphId) {
    return graph();
  } else {
    return dv_p->lowering().getVirtualGraph(vgid.first, vgid.second);
  }
}

snap::Graph &PopOpx::dstVirtualGraph(OutIndex index) const {
  auto &op  = getOp<Op>();
  auto vgid = op.getIntrospectionOutVirtualGraphId(index);
  if (vgid.first == unusedVGraphId) {
    return graph();
  } else {
    return dv_p->lowering().getVirtualGraph(vgid.first, vgid.second);
  }
}

const poplar::Tensor &PopOpx::get(TensorId id) const {
  return dv_p->lowering().tensors().get(id);
}

const poplar::Tensor &PopOpx::getView(TensorId id) const {
  return dv_p->lowering().tensors().getView(id);
}

void PopOpx::insert(TensorId id, const poplar::Tensor &tensor) const {
  dv_p->lowering().tensors().insert(id, tensor);
}

TensorId PopOpx::inId(InIndex index) const { return op_p->input->id(index); }
TensorId PopOpx::outId(OutIndex index) const { return op_p->output->id(index); }

bool PopOpx::hasInput(InIndex index) const {
  return op_p->input->hasIndex(index);
}

bool PopOpx::hasOutput(OutIndex index) const {
  return op_p->output->hasIndex(index);
}

const poplar::Tensor &PopOpx::getInTensor(InIndex index) const {
  if (!cachedInputs.empty()) {
    return cachedInputs[index];
  } else {
    return get(op_p->input->id(index));
  }
}

const poplar::Tensor &PopOpx::getOutTensor(OutIndex index) const {
  if (cachedOutputs && !cachedOutputs->empty()) {
    return (*cachedOutputs)[index];
  } else {
    return get(op_p->output->id(index));
  }
}

const poplar::Tensor &PopOpx::getInView(InIndex index) const {
  return getView(op_p->input->id(index));
}

const poplar::Tensor &PopOpx::getOutView(OutIndex index) const {
  return getView(op_p->output->id(index));
}

bool PopOpx::hasInViewChangers(InIndex index) const {
  return dv_p->lowering().tensors().hasViewChangers(op_p->input->id(index));
}

const ViewChangers &PopOpx::getInViewChangers(InIndex index) const {
  return dv_p->lowering().tensors().getViewChangers(op_p->input->id(index));
}

void PopOpx::setOutViewChangers(OutIndex index,
                                const ViewChangers &changers) const {
  return dv_p->lowering().tensors().setViewChangers(op_p->output->id(index),
                                                    changers);
}

void PopOpx::setOutTensor(OutIndex index, const poplar::Tensor &tensor) const {

  if (dv_p->lowering().ir().getSessionOptions().opxAliasChecking) {
    // Verify no unsolicited aliasing takes place
    Op &op = getOp<Op>();
    for (auto inputs : op.input->indicesMap()) {
      InIndex inIndex   = inputs.second.front();
      auto &inputTensor = getInTensor(inIndex);
      if (tensor.elementType() == inputTensor.elementType()) {
        if (!tensor.containsAliases() && !inputTensor.containsAliases()) {
          // Can only safely test for aliases between the input and output
          // if the input and output tensors alone are free from aliases
          auto aliasedRegions = op.aliases(inIndex, index);
          bool aliasesInIr =
              std::any_of(aliasedRegions.begin(),
                          aliasedRegions.end(),
                          [](view::Region &r) { return !r.isEmpty(); });
          bool aliasesInPoplar =
              poplar::concat(tensor.flatten(), inputTensor.flatten(), 0)
                  .containsAliases();
          // If the op is outplace in the ir, but inplace in poplar, that is an
          // error. Note, there may be cases where the op is inplace in the ir,
          // but the poplar operation was not able to be inplaced, hence why we
          // do not just compare `aliasesInIr != aliasesInPoplar`.
          if (!aliasesInIr && aliasesInPoplar) {
            throw error(
                "Op {} claims input {} -> output {} {} contain aliases, "
                "but the Poplar tensors disagree.",
                op.debugName(),
                inIndex,
                index,
                aliasesInIr ? "do" : "do not");
          }
        }
      }
    }
  }

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

Tensor *PopOpx::inTensor(InIndex index) const {
  return op_p->input->tensor(index);
}
Tensor *PopOpx::outTensor(OutIndex index) const {
  return op_p->output->tensor(index);
}

const TensorInfo &PopOpx::inInfo(InIndex index) const {
  return inTensor(index)->info;
}

const Shape &PopOpx::inShape(InIndex index) const {
  return inInfo(index).shape();
}

const TensorInfo &PopOpx::outInfo(OutIndex index) const {
  return outTensor(index)->info;
}

const Shape &PopOpx::outShape(OutIndex index) const {
  return outInfo(index).shape();
}

// If the operator has been named return the name, (i.e. "my_add/23")
// else return the id (i.e "23")
std::string PopOpx::idStr() const {
  if (!op_p->name().empty()) {
    return op_p->name() + sNameDelimiter + std::to_string(op_p->id);
  } else {
    return std::to_string(op_p->id);
  }
}

const popart::DebugInfo &PopOpx::getDebugInfo() const {
  return op_p->getDebugInfo();
}

const poplar::DebugNameAndId
PopOpx::getDebugNameAndId(const std::string name,
                          poplar::SourceLocation loc) const {
  auto &di = getDebugInfo();
  return poplar::DebugNameAndId(name, di.getId(), di.getPathName());
}

poplar::DebugContext PopOpx::debugContext(const std::string name,
                                          poplar::SourceLocation loc) const {
  return {getDebugNameAndId(), name, loc};
}

poplar::Tensor PopOpx::cloneNcopy(poplar::program::Sequence &prog,
                                  TensorId id) const {
  const poplar::Tensor &tensor = get(id);
  return cloneNcopy(prog, tensor, id + "[cloned]");
}

poplar::Tensor PopOpx::cloneNcopy(poplar::program::Sequence &prog,
                                  const poplar::Tensor &tensor,
                                  const std::string name) const {

  const auto scopedTimer =
      getDevicex()->ir().timePartitionLogger().scopedStopwatch(
          "Clone (and copy)");

  // TODO Would be good to get the name of the tensor
  auto outTensor = graph().getPoplarGraph().clone(tensor, debugContext(name));
  prog.add(poplar::program::Copy(tensor, outTensor, false, debugContext()));
  return outTensor;
}

poplar::Tensor PopOpx::broadcast(const std::vector<int64_t> &desired_shape,
                                 TensorId id) const {
  return broadcast(desired_shape, get(id));
}

const Devicex *PopOpx::getDevicex() const { return dv_p; }

poplar::Tensor PopOpx::broadcast(const std::vector<int64_t> &desired_shape,
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

poplar::Tensor PopOpx::getConst(const poplar::Type &type,
                                const std::vector<size_t> &shape,
                                double val,
                                const std::string &name) const {
  return dv_p->lowering().getConst(
      graph(), type, shape, val, debugContext(name));
}

poplar::Tensor PopOpx::getScalarVariable(const poplar::Type &type,
                                         const std::string &name) const {
  return dv_p->lowering().getScalarVariable(graph(), type, debugContext(name));
}

std::vector<std::tuple<TensorId, TensorId, bool>>
PopOpx::getOutputsToPrepare() const {
  return {};
}

std::vector<std::tuple<TensorId, TensorId, bool>>
PopOpx::getInputsToPrepare() const {
  return {};
}

} // namespace popx
} // namespace popart
