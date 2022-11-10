// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "popart/popx/debugcontextx.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <poprithms/logging/timepartitionlogger.hpp>

#include "popart/error.hpp"
#include "popart/graph.hpp"
#include "popart/ir.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/popx/devicex.hpp"
#include "popart/popx/irlowering.hpp"
#include "popart/popx/opx.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/region.hpp"
#include "popart/subgraphpartitioner.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
namespace popx {

Opx::Opx(Op *op_p_, Devicex *dv_p_) : op_p(op_p_), dv_p(dv_p_) {}
Opx::~Opx() = default;

poplar::Tensor Opx::createInput(InIndex index,
                                const poplar::DebugNameAndId &dnai) const {
  throw error("Opx for {} cannot create Input index:{} name:{}",
              op_p->opid,
              index,
              dnai.getPathName());
}

snap::Tensor Opx::createInputTensor(InIndex index,
                                    const poplar::DebugNameAndId &dnai) const {
  return snap::Tensor{createInput(index, dnai), snapGraph()};
}

std::set<TensorId> Opx::mustExistBeforeCreate(int index0) const {
  throw error("Opx for {} cannot say which poplar Tensors must exist to create "
              "at index {}",
              op_p->opid,
              index0);
}

DnfTensorIds Opx::mustExistBeforeCreateDNF(int index0) const {
  return {mustExistBeforeCreate(index0)};
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
Opx::unwindTensorLayout(poplar::Tensor tensor, InIndex in, OutIndex out) const {
  throw error("Opx for {} cannot unwind the tensor layout change between input "
              "and output for {}",
              op_p->opid);
}

snap::Tensor
Opx::unwindTensorLayout(snap::Tensor tensor, InIndex in, OutIndex out) const {
  return snap::Tensor{unwindTensorLayout(tensor.getPoplarTensor(), in, out),
                      snapGraph()};
}

bool Opx::createsEquiv(int, const Opx *, int) const {
  throw error("No check for equivalent tensor create for type {}", op_p->opid);
}

view::RegMap Opx::unwindRegion(InIndex, OutIndex) const {
  throw error("Opx cannot unwind the region between input "
              "and output for {}",
              op_p->opid);
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
    return dv_p->lowering()
        .getVirtualGraph(getVirtualGraphId(), op_p->settings.tileSet)
        .getPoplarGraph();
  } else {
    return dv_p->lowering().graph().getPoplarGraph();
  }
}

const poplar::Tensor &Opx::get(TensorId id) const {
  return dv_p->lowering().tensors().get(id).getPoplarTensor();
}

const poplar::Tensor &Opx::getView(TensorId id) const {
  return dv_p->lowering().tensors().getView(id).getPoplarTensor();
}

void Opx::insert(TensorId id, const poplar::Tensor &tensor) const {
  dv_p->lowering().tensors().insert(
      id, snap::Tensor{tensor, dv_p->lowering().graph()});
}

TensorId Opx::inId(InIndex index) const { return op_p->input->id(index); }
TensorId Opx::outId(OutIndex index) const { return op_p->output->id(index); }

const poplar::Tensor &Opx::getInTensor(InIndex index) const {

  return get(op_p->input->id(index));
}

const poplar::Tensor &Opx::getOutTensor(OutIndex index) const {

  return get(op_p->output->id(index));
}

const poplar::Tensor &Opx::getInView(InIndex index) const {
  return getView(op_p->input->id(index));
}

const poplar::Tensor &Opx::getOutView(OutIndex index) const {
  return getView(op_p->output->id(index));
}

bool Opx::hasInViewChangers(InIndex index) const {
  return dv_p->lowering().tensors().hasViewChangers(op_p->input->id(index));
}

const ViewChangers &Opx::getInViewChangers(InIndex index) const {
  return dv_p->lowering().tensors().getViewChangers(op_p->input->id(index));
}

void Opx::setOutViewChangers(OutIndex index,
                             const ViewChangers &changers) const {
  return dv_p->lowering().tensors().setViewChangers(op_p->output->id(index),
                                                    changers);
}

void Opx::setOutTensor(OutIndex index, const poplar::Tensor &tensor) const {

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

  logging::trace("Op {} inserting poplar::Tensor {}",
                 getOp<Op>().debugName(),
                 op_p->output->id(index));
  insert(op_p->output->id(index), tensor);
}

bool Opx::hasInput(InIndex index) const { return op_p->input->hasIndex(index); }

bool Opx::hasOutput(OutIndex index) const {
  return op_p->output->hasIndex(index);
}

const Devicex *Opx::getDevicex() const { return dv_p; }

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

const std::vector<size_t> Opx::inShapeSzt(InIndex index) const {
  return inInfo(index).shape_szt();
}

const TensorInfo &Opx::outInfo(OutIndex index) const {
  return outTensor(index)->info;
}

const Shape &Opx::outShape(OutIndex index) const {
  return outInfo(index).shape();
}

poplar::Tensor Opx::getConst(const poplar::Type &type,
                             const std::vector<size_t> &shape,
                             double val,
                             const std::string &name) const {
  return dv_p->lowering()
      .getConst(snapGraph(), type, shape, val, debugContext(name))
      .getPoplarTensor();
}

poplar::Tensor Opx::getScalarVariable(const poplar::Type &type,
                                      const std::string &name) const {
  return dv_p->lowering()
      .getScalarVariable(snapGraph(), type, debugContext(name))
      .getPoplarTensor();
}

PreparedTensorInfos Opx::getOutputsToPrepare() const { return {}; }

PreparedTensorInfos Opx::getInputsToPrepare() const { return {}; }

std::set<OpxGrowPartId> Opx::getInGrowPartIds(Tensor *inTensor) const {
  // By default, growing in parts is disabled
  return {};
}

OpxGrowPartId Opx::getOutGrowPartId(Tensor *outTensor) const {
  // By default, growing in parts is disabled
  return unusedGrowPartId;
}

void Opx::grow(snap::program::Sequence &prog) const {
  grow(prog.getPoplarSequence());
}

void Opx::grow(poplar::program::Sequence &) const {
  throw error("adding poplar::Tensors not implemented for {}", op_p->opid);
}

void Opx::grow(std::vector<snap::program::Sequence> &sequences) const {
  if (sequences.empty()) {
    auto partitioner  = dv_p->lowering().getSubgraphPartitioner();
    auto subgraphPart = partitioner->getOpSubgraphPartBegin(op_p);

    std::stringstream ss;
    ss << op_p->getGraph().id.str() << "/" << subgraphPart;
    sequences.resize(1,
                     snap::program::Sequence(
                         debugContext(ss.str()),
                         // Using `graph()` here was causing an error due
                         // to ipucopy not having a virtual graph id set.
                         // snap::program::Sequence does not require a specific
                         // graph though, just any snap::Graph, so use the main
                         // graph provided by `dv_p->lowering().graph()`.
                         dv_p->lowering().graph()));
  }

  // By default, use the Opx::grow(snap::program::Sequence &) function.
  // Currently, only CallOpx overloads this Opx::grow method to grow over
  // multiple fragments.
  grow(*sequences.begin());
}

void Opx::grow(std::vector<poplar::program::Sequence> &sequences) const {
  if (sequences.empty()) {
    auto partitioner  = dv_p->lowering().getSubgraphPartitioner();
    auto subgraphPart = partitioner->getOpSubgraphPartBegin(op_p);

    std::stringstream ss;
    ss << op_p->getGraph().id.str() << "/" << subgraphPart;
    sequences.resize(1,
                     snap::program::Sequence(
                         debugContext(ss.str()),
                         // Using `graph()` here was causing an error due
                         // to ipucopy not having a virtual graph id set.
                         // snap::program::Sequence does not require a specific
                         // graph though, just any snap::Graph, so use the main
                         // graph provided by `dv_p->lowering().graph()`.
                         dv_p->lowering().graph()));
  }

  // By default, use the Opx::grow(poplar::program::Sequence &) function.
  // Currently, only CallOpx overloads this Opx::grow method to grow over
  // multiple fragments.
  grow(*sequences.begin());
}

void Opx::growPart(OpxGrowPartId id) const {
  // By default, growing in parts is disabled
  throw error("part growing not implemented for {}", op_p->opid);
}

bool Opx::hasCreatorViewChangers(InIndex) const { return false; }

ViewChangers Opx::getCreatorViewChangers(InIndex) const {
  return ViewChangers{};
}

const popart::DebugInfo &Opx::getDebugInfo() const {
  return op_p->getDebugInfo();
}

const poplar::DebugNameAndId
Opx::getDebugNameAndId(const std::string name,
                       poplar::SourceLocation loc) const {
  auto &di = getDebugInfo();
  return poplar::DebugNameAndId(name, di.getId(), di.getPathName());
}

poplar::DebugContext Opx::debugContext(const std::string name,
                                       poplar::SourceLocation loc) const {
  return {getDebugNameAndId(), name, loc};
}

const snap::Tensor &Opx::snapGet(TensorId id) const {
  return dv_p->lowering().tensors().get(id);
}

snap::Graph &Opx::snapGraph() const {
  if (op_p->getIr().virtualGraphsEnabled()) {
    return dv_p->lowering().getVirtualGraph(getVirtualGraphId(),
                                            op_p->settings.tileSet);
  } else {
    return dv_p->lowering().graph();
  }
}

snap::Graph &Opx::snapSrcVirtualGraph(InIndex index) const {
  auto &op  = getOp<Op>();
  auto vgid = op.getIntrospectionInVirtualGraphId(index);
  if (vgid.first == unusedVGraphId) {
    return snapGraph();
  } else {
    return dv_p->lowering().getVirtualGraph(vgid.first, vgid.second);
  }
}

const snap::Tensor &Opx::snapGetInTensor(InIndex index) const {

  return dv_p->lowering().tensors().get(op_p->input->id(index));
}

const snap::Tensor &Opx::snapGetOutTensor(OutIndex index) const {

  return dv_p->lowering().tensors().get(op_p->output->id(index));
}

} // namespace popx
} // namespace popart
