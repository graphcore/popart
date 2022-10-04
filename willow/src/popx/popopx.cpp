// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popart/popx/debugcontextx.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <utility>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <poprithms/logging/timepartitionlogger.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/viewchangers.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include "popart/debugcontext.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/opdebuginfo.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/popx/poptensors.hpp"
#include "popart/popx/preparedtensor.hpp"
#include "popart/region.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/subgraphpartitioner.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
namespace popx {

PopOpx::PopOpx(Op *op_p_, Devicex *dv_p_) : op_p(op_p_), dv_p(dv_p_) {}

PopOpx::~PopOpx() = default;

snap::Tensor
PopOpx::createInputTensor(InIndex index,
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

void PopOpx::grow(snap::program::Sequence &) const {
  throw error("adding poplar::Tensors not implemented for {}", op_p->opid);
}

void PopOpx::grow(std::vector<snap::program::Sequence> &sequences) const {
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

  // By default, use the PopOpx::grow(snap::program::Sequence &) function.
  // Currently, only CallOpx overloads this PopOpx::grow method to grow over
  // multiple fragments.
  grow(*sequences.begin());
}

std::set<OpxGrowPartId> PopOpx::getInGrowPartIds(Tensor *inTensor) const {
  // By default, growing in parts is disabled
  return {};
}

OpxGrowPartId PopOpx::getOutGrowPartId(Tensor *outTensor) const {
  // By default, growing in parts is disabled
  return unusedGrowPartId;
}

void PopOpx::growPart(OpxGrowPartId id) const {
  // By default, growing in parts is disabled
  throw error("part growing not implemented for {}", op_p->opid);
}

InputCreatorType PopOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::Deadend;
}

bool PopOpx::canUnwind(InIndex in, OutIndex) const {
  auto type = getInputCreatorType(in);
  return type == InputCreatorType::CanUnwind ||
         type == InputCreatorType::CanCreateOrUnwind;
}

snap::Tensor PopOpx::unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const {
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

snap::Graph &PopOpx::topLevelGraph() const { return dv_p->lowering().graph(); }

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

const snap::Tensor &PopOpx::get(TensorId id) const {
  return dv_p->lowering().tensors().get(id);
}

const snap::Tensor &PopOpx::getView(TensorId id) const {
  return dv_p->lowering().tensors().getView(id);
}

void PopOpx::insert(TensorId id, const snap::Tensor &tensor) const {
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

const snap::Tensor &PopOpx::getInTensor(InIndex index) const {
  if (!cachedInputs.empty()) {
    return cachedInputs[index];
  } else {
    return get(op_p->input->id(index));
  }
}

const snap::Tensor &PopOpx::getOutTensor(OutIndex index) const {
  if (cachedOutputs && !cachedOutputs->empty()) {
    return (*cachedOutputs)[index];
  } else {
    return get(op_p->output->id(index));
  }
}

const snap::Tensor &PopOpx::getInView(InIndex index) const {
  return getView(op_p->input->id(index));
}

const snap::Tensor &PopOpx::getOutView(OutIndex index) const {
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

void PopOpx::setOutTensor(OutIndex index, const snap::Tensor &t) const {
  if (dv_p->lowering().ir().getSessionOptions().opxAliasChecking) {
    // Verify no unsolicited aliasing takes place
    Op &op = getOp<Op>();
    for (auto inputs : op.input->indicesMap()) {
      InIndex inIndex   = inputs.second.front();
      auto &inputTensor = getInTensor(inIndex);
      if (t.elementType() == inputTensor.elementType()) {
        if (!t.getPoplarTensor().containsAliases() &&
            !inputTensor.getPoplarTensor().containsAliases()) {
          // Can only safely test for aliases between the input and output
          // if the input and output tensors alone are free from aliases
          auto aliasedRegions = op.aliases(inIndex, index);
          bool aliasesInIr =
              std::any_of(aliasedRegions.begin(),
                          aliasedRegions.end(),
                          [](view::Region &r) { return !r.isEmpty(); });
          bool aliasesInPoplar =
              snap::concat(t.flatten(), inputTensor.flatten(), 0)
                  .getPoplarTensor()
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
    cachedOutputs->insert(cachedOutputs->begin() + index, t);
  } else {
    logging::trace("Op {} inserting poplar::Tensor {}",
                   getOp<Op>().debugName(),
                   op_p->output->id(index));
    insert(op_p->output->id(index), t);
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

const std::vector<size_t> PopOpx::inShapeSzt(InIndex index) const {
  return inInfo(index).shape_szt();
}

const TensorInfo &PopOpx::outInfo(OutIndex index) const {
  return outTensor(index)->info;
}

const Shape &PopOpx::outShape(OutIndex index) const {
  return outInfo(index).shape();
}

const std::vector<size_t> PopOpx::outShapeSzt(OutIndex index) const {
  return outInfo(index).shape_szt();
}

snap::Graph &PopOpx::inGraph(InIndex in) const {
  if (op_p->hasVirtualGraphId()) {
    std::set<OpId> visited;
    auto vgid = op_p->getIntrospectionInVirtualGraphId(in, visited);
    return dv_p->lowering().getVirtualGraph(vgid.first, vgid.second);
  }
  return dv_p->lowering().graph();
}

snap::Graph &PopOpx::outGraph(OutIndex out) const {
  if (op_p->hasVirtualGraphId()) {
    std::set<OpId> visited;
    auto vgid = op_p->getIntrospectionOutVirtualGraphId(out, visited);
    return dv_p->lowering().getVirtualGraph(vgid.first, vgid.second);
  }
  return dv_p->lowering().graph();
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

snap::Tensor PopOpx::cloneNcopy(snap::program::Sequence &prog,
                                TensorId id) const {
  const snap::Tensor &tensor = get(id);
  return cloneNcopy(prog, tensor, id + "[cloned]");
}

snap::Tensor PopOpx::cloneNcopy(snap::program::Sequence &prog,
                                const snap::Tensor &tensor,
                                const std::string name) const {

  const auto scopedTimer =
      getDevicex()->ir().timePartitionLogger().scopedStopwatch(
          "Clone (and copy)");

  // TODO Would be good to get the name of the tensor
  auto outTensor = graph().clone(tensor, debugContext(name));
  prog.getPoplarSequence().add(
      snap::program::Copy(tensor, outTensor, false, debugContext(name)));
  return outTensor;
}

const Devicex *PopOpx::getDevicex() const { return dv_p; }

snap::Tensor PopOpx::getConst(const poplar::Type &type,
                              const std::vector<size_t> &shape,
                              double val,
                              const std::string &name) const {
  return dv_p->lowering().getConst(
      graph(), type, shape, val, debugContext(name));
}

snap::Tensor PopOpx::getScalarVariable(const poplar::Type &type,
                                       const std::string &name) const {
  return dv_p->lowering().getScalarVariable(graph(), type, debugContext(name));
}

snap::Tensor PopOpx::getZerosTensor(std::vector<std::size_t> shape,
                                    poplar::Type elem_type,
                                    std::string name = "") const {
  // create scalar variable with provided elem_type and name
  auto zero = getScalarVariable(elem_type, name).getPoplarTensor();
  // set the variable's value to 0
  graph().getPoplarGraph().setInitialValue(zero, 0);
  // broadcast variable to required shape
  for (int i = shape.size() - 1; i >= 0; i--) {
    zero = zero.expand({0});
    zero = zero.broadcast(shape[i], 0);
  }
  return snap::Tensor{zero, graph()};
}

PreparedTensorInfos PopOpx::getOutputsToPrepare() const { return {}; }

PreparedTensorInfos PopOpx::getInputsToPrepare() const { return {}; }

} // namespace popx
} // namespace popart
