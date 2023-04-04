// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/range/algorithm/copy.hpp>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <onnx/onnx_pb.h>
#include <onnxutil.hpp>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <popart/bwdgraphinfo.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/if.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/transforms/autodiff/calledgraphgradophelper.hpp>
#include <popart/util.hpp>

#include "popart/analysis/replicaequal/replicaequalanalysisproxy.hpp"
#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/identity.hpp"
#include "popart/operators.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"
#include <boost/container/flat_set.hpp>

namespace popart {
class AliasModel;
enum class RequireOptimalSchedule;
struct OperatorIdentifier;
template <typename T> using FlatSet = boost::container::flat_set<T>;

namespace {

SubgraphIndex thenSubgraphIndex = 0;
SubgraphIndex elseSubgraphIndex = 1;

} // namespace

std::map<int, int> IfOp::getInIndicesMapForGradOp(
    const std::map<TensorId, int> &opInIdToOpInIdx,
    const std::map<TensorId, int> &opInIdToGraphInIdx) const {
  std::map<int, int> result;
  for (auto id_idx : opInIdToGraphInIdx) {
    auto opInId     = id_idx.first;
    auto graphInIdx = id_idx.second;
    auto opInIdx    = opInIdToOpInIdx.at(opInId);
    result.insert({opInIdx, graphInIdx});
  }
  return result;
}

std::map<int, int> IfOp::getOutIndicesMapForGradOp(
    const std::map<InIndex, InIndex> &idxMap) const {

  std::map<int, int> indicesMap;
  for (int i = 0; i < input->n(); i++) {
    // fwdBranch input index == bwdBranch output index
    auto found = idxMap.find(i);
    if (found != idxMap.end()) {
      auto branchInputIndex = found->second;
      // fwdOp input index == bwdOp output index - 1
      // -1 as there is no condition output
      indicesMap.insert({i - 1, branchInputIndex});
    }
  }
  return indicesMap;
}

BranchInfo::BranchInfo(const GraphId &graphId_,
                       const std::map<int, int> inputIndicesMap_,
                       const std::map<int, int> outputIndicesMap_)
    : graphId(graphId_), inputIndicesMap(inputIndicesMap_),
      outputIndicesMap(outputIndicesMap_) {}

IfOp::IfOp(const OperatorIdentifier &opid_,
           const BranchInfo &thenBranchInfo,
           const BranchInfo &elseBranchInfo,
           const Op::Settings &settings_)
    : Op(opid_, settings_), thenInputIndicesMap(thenBranchInfo.inputIndicesMap),
      elseInputIndicesMap(elseBranchInfo.inputIndicesMap),
      thenOutputIndicesMap(thenBranchInfo.outputIndicesMap),
      elseOutputIndicesMap(elseBranchInfo.outputIndicesMap),
      thenGraphId(thenBranchInfo.graphId),
      elseGraphId(elseBranchInfo.graphId), calledGraphGradOpHelper{this} {}

std::unique_ptr<Op> IfOp::clone() const {
  return std::make_unique<IfOp>(*this);
}

std::map<TensorId, TensorId> IfOp::getBranchOutIdToOpOutIdMap() const {
  std::map<TensorId, TensorId> result;

  auto addResultsForGraph = [this, &result](const Graph &graph) {
    auto &idxMap = getBranchOutIndicesMap(graph);
    for (auto opIdx_branchIdx : idxMap) {
      auto opIdx       = opIdx_branchIdx.first;
      auto branchIdx   = opIdx_branchIdx.second;
      auto opOutId     = outId(opIdx);
      auto branchOutId = graph.getOutputId(branchIdx);
      result.insert({branchOutId, opOutId});
    }
  };

  addResultsForGraph(getThenGraph());
  addResultsForGraph(getElseGraph());

  return result;
}

std::vector<TensorId> IfOp::getGradOpInputIds(const Graph &gradThenGraph,
                                              const Graph &gradElseGraph) {
  using boost::range::copy;

  auto branchOutIdToOpOutId = getBranchOutIdToOpOutIdMap();

  std::set<TensorId> requiredGradOpInputs;

  auto addInputTensors = [&](const Graph &fwdGraph,
                             const Graph &bwdGraph,
                             const BwdGraphInfo &gradInfo) {
    for (InIndex i = 0; i < gradInfo.expectedInputs.size(); ++i) {
      auto &expIn = gradInfo.expectedInputs.at(i);
      switch (expIn.type) {
      case ExpectedConnectionType::Fwd: {
        auto scopedId   = expIn.fwdId;
        auto unscopedId = removeScope(fwdGraph, scopedId);
        requiredGradOpInputs.insert(unscopedId);
        break;
      }
      case ExpectedConnectionType::FwdGrad: {
        auto scopedId  = expIn.fwdId;
        auto opOutIdIt = branchOutIdToOpOutId.find(scopedId);
        if (opOutIdIt != branchOutIdToOpOutId.end()) {
          auto opOutId = opOutIdIt->second;
          auto gradId  = getGradId(opOutId);
          requiredGradOpInputs.insert(gradId);
        } else {
          throw error("[IfOp::getGradOpInputIds] Expected the forward tensor "
                      "'{}' of {} (the gradient of which is a graph input, "
                      "'{}', of {}) to be a graph output of {}",
                      scopedId,
                      fwdGraph.getGraphString(),
                      bwdGraph.getInputId(i),
                      bwdGraph.getGraphString(),
                      fwdGraph.getGraphString());
        }
        break;
      }
      default:
        throw error("Unsupported ExpectedConnectionType");
      }
    }
  };

  const auto &calledGraphsGradInfo =
      calledGraphGradOpHelper.getCalledSubgraphGradInfo();
  auto &thenGradInfo = calledGraphsGradInfo.at(getThenGraph().id);
  auto &elseGradInfo = calledGraphsGradInfo.at(getElseGraph().id);
  addInputTensors(getThenGraph(), gradThenGraph, thenGradInfo);
  addInputTensors(getElseGraph(), gradElseGraph, elseGradInfo);

  // condition tensor must be first
  std::vector<TensorId> result{inId(getConditionInIndex())};
  copy(requiredGradOpInputs, std::back_inserter(result));
  return result;
}

std::map<TensorId, int>
IfOp::getOpInIdToBwdGraphInIndexMap(const Graph &fwdGraph,
                                    const Graph &bwdGraph) const {
  auto branchOutIdToOpOutId = getBranchOutIdToOpOutIdMap();
  const auto &calledGraphsGradInfo =
      calledGraphGradOpHelper.getCalledSubgraphGradInfo();
  auto &gradInfo = calledGraphsGradInfo.at(fwdGraph.id);

  std::map<TensorId, int> result;
  for (InIndex i = 0; i < gradInfo.expectedInputs.size(); ++i) {
    auto &expIn = gradInfo.expectedInputs.at(i);
    switch (expIn.type) {
    case ExpectedConnectionType::Fwd: {
      // branch input to tensor id
      auto branchInId = expIn.fwdId;
      auto opInId     = removeScope(fwdGraph, branchInId);
      result.insert({opInId, i});
      break;
    }
    case ExpectedConnectionType::FwdGrad: {
      auto branchOutId = expIn.fwdId;
      auto opOutIdIt   = branchOutIdToOpOutId.find(branchOutId);
      if (opOutIdIt != branchOutIdToOpOutId.end()) {
        auto opOutId = opOutIdIt->second;
        auto gradId  = getGradId(opOutId);
        result.insert({gradId, i});
      } else {
        throw error("[IfOp::getGradOpInputIds] Expected the forward tensor "
                    "'{}' of {} (the gradient of which is a graph input, "
                    "'{}', of {}) to be a graph output of {}",
                    branchOutId,
                    fwdGraph.getGraphString(),
                    bwdGraph.getInputId(i),
                    bwdGraph.getGraphString(),
                    fwdGraph.getGraphString());
      }
      break;
    }
    default:
      throw error("Unsupported ExpectedConnectionType");
    }
  }

  return result;
}

std::vector<GradInOutMapper>
IfOp::getGradInInfo(const std::vector<TensorId> &gradOpInputIds) const {
  std::vector<GradInOutMapper> gradInInfo;

  auto tryAddInput = [this, &gradInInfo](InIndex gradOpInIdx,
                                         const TensorId &gradOpInId) {
    for (auto &idx_tensor : input->tensorMap()) {
      int idx     = idx_tensor.first;
      auto tensor = idx_tensor.second;
      if (gradOpInId == tensor->id) {
        gradInInfo.push_back({gradOpInIdx, idx, GradOpInType::In});
        return true;
      }
    }
    return false;
  };

  auto tryAddGrad = [this, &gradInInfo](InIndex gradOpInIdx,
                                        const TensorId &gradOpInId) {
    for (auto &idx_tensor : output->tensorMap()) {
      int idx     = idx_tensor.first;
      auto tensor = idx_tensor.second;
      auto gradId = getGradId(tensor->id);
      if (gradOpInId == gradId) {
        gradInInfo.push_back({gradOpInIdx, idx, GradOpInType::GradOut});
        return true;
      }
    }
    return false;
  };

  for (int gradOpInputIdx = 0; gradOpInputIdx < gradOpInputIds.size();
       gradOpInputIdx++) {
    auto gradOpInputId = gradOpInputIds.at(gradOpInputIdx);
    if (tryAddInput(gradOpInputIdx, gradOpInputId)) {
    } else if (tryAddGrad(gradOpInputIdx, gradOpInputId)) {
    } else {
      throw error("Could not add grad input info for tensor {}", gradOpInputId);
    }
  }

  return gradInInfo;
}

BranchInfo
IfOp::getBwdGraphBranchInfo(const Graph &fwdGraph,
                            const Graph &bwdGraph,
                            const std::vector<TensorId> &gradOpInputIds) const {
  // get a map of IfGradOp input ids to IfGradOp input indices
  std::map<TensorId, int> gradOpInIdToGradOpInIdx;
  for (int i = 0; i < gradOpInputIds.size(); i++) {
    auto inId = gradOpInputIds.at(i);
    gradOpInIdToGradOpInIdx.insert({inId, i});
  }

  // get a map of IfGradOp input ids to bwdGraph input indices
  auto gradOpInIdToBwdGraphInIdx =
      getOpInIdToBwdGraphInIndexMap(fwdGraph, bwdGraph);

  auto bwdInputIndicesMap = getInIndicesMapForGradOp(gradOpInIdToGradOpInIdx,
                                                     gradOpInIdToBwdGraphInIdx);

  auto fwdInputIndicesMap  = getBranchInIndicesMap(fwdGraph);
  auto bwdOutputIndicesMap = getOutIndicesMapForGradOp(fwdInputIndicesMap);

  return {bwdGraph.id, bwdInputIndicesMap, bwdOutputIndicesMap};
}

std::vector<std::unique_ptr<Op>> IfOp::getGradOps() {
  auto &bwdThenGraph = calledGraphGradOpHelper.getBwdGraph(thenSubgraphIndex);
  auto &bwdElseGraph = calledGraphGradOpHelper.getBwdGraph(elseSubgraphIndex);

  auto gradOpInputIds = getGradOpInputIds(bwdThenGraph, bwdElseGraph);
  auto bwdThenBranchInfo =
      getBwdGraphBranchInfo(getThenGraph(), bwdThenGraph, gradOpInputIds);
  auto bwdElseBranchInfo =
      getBwdGraphBranchInfo(getElseGraph(), bwdElseGraph, gradOpInputIds);

  auto gradInInfo = getGradInInfo(gradOpInputIds);

  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<IfGradOp>(
      *this, gradInInfo, bwdThenBranchInfo, bwdElseBranchInfo));
  upops.emplace_back(std::make_unique<IfConditionGradOp>(*this));
  return upops;
}

const std::map<InIndex, InIndex> &
IfOp::getBranchInIndicesMap(const Graph &branchGraph) const {
  if (&branchGraph == &getThenGraph()) {
    return thenInputIndicesMap;
  } else if (&branchGraph == &getElseGraph()) {
    return elseInputIndicesMap;
  } else {
    throw error("Graph {} is not a branch of IfOp", branchGraph.id);
  }
}

const std::map<OutIndex, OutIndex> &
IfOp::getBranchOutIndicesMap(const Graph &branchGraph) const {
  if (&branchGraph == &getThenGraph()) {
    return thenOutputIndicesMap;
  } else if (&branchGraph == &getElseGraph()) {
    return elseOutputIndicesMap;
  } else {
    throw error("Graph {} is not a branch of IfOp", branchGraph.id);
  }
}

std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
IfOp::fwdPropagateIsReplicaEqual(const AliasModel &aliasModel,
                                 const ReplEqInputMap &opInputMap,
                                 ReplicaEqualAnalysisProxy &proxy) const {

  auto getOpOutputMap = [&](const Graph *subgraph,
                            SubgraphIndex subgraphIndex) {
    // Map Op inputs mapping to subgraph input mapping.
    ReplEqInputMap subgraphInputMap;
    for (InIndex i = 0; i < subgraph->getInputIds().size(); ++i) {
      InIndex opInIndex   = subgraphInToOpInIndex(subgraphIndex, i);
      subgraphInputMap[i] = opInputMap.at(opInIndex);
    }

    // Forward propagate on subgraph.
    auto subgraphRes = proxy.fwdPropagateIsReplicaEqualThroughGraph(
        subgraph, subgraphInputMap);
    auto &subgraphOutputMap = std::get<0>(subgraphRes);

    // Map Op output mapping back to subgraph output mapping.
    ReplEqInputMap opOutputMap;
    for (auto &entry : output->tensorMap()) {
      OutIndex opOutIndex = entry.first;
      OutIndex subgraphOutIndex =
          opOutToSubgraphOutIndex(subgraphIndex, opOutIndex);
      opOutputMap[opOutIndex] = subgraphOutputMap.at(subgraphOutIndex);
    }

    return opOutputMap;
  };

  // Get the output map for each subgraph independently.
  auto thenOpOutputMap = getOpOutputMap(&getThenGraph(), thenSubgraphIndex);
  auto elseOpOutputMap = getOpOutputMap(&getElseGraph(), elseSubgraphIndex);

  // Merge the results, we can only guarantee an output is replica equal if it's
  // replica equal for both subgraphs.
  ReplEqInputMap opOutputMap;
  for (auto &entry : output->tensorMap()) {
    OutIndex opOutIndex = entry.first;
    opOutputMap[opOutIndex] =
        thenOpOutputMap.at(opOutIndex) && elseOpOutputMap.at(opOutIndex);
  }

  // At the moment, it is not possible to modify an input with an IfOp, so we
  // return an empty modified inputs value list.
  return {opOutputMap, {}};
}

void IfOp::setup() {

  auto trySetOutInfoFromGraph = [this](const Graph &graph, int outIndex) {
    auto &idxMap = getBranchOutIndicesMap(graph);
    auto found   = idxMap.find(outIndex);
    if (found != idxMap.end()) {
      auto branchId     = graph.getOutputId(found->second);
      auto branchTensor = graph.getTensors().get(branchId);
      outInfo(outIndex) = branchTensor->info;
      return true;
    } else {
      return false;
    }
  };

  for (int i = 0; i < output->n(); i++) {
    if (!trySetOutInfoFromGraph(getThenGraph(), i) &&
        !trySetOutInfoFromGraph(getElseGraph(), i)) {
      throw error(
          "Could not find suitable branch output for IfGradOp output {}", i);
    }
  }
}

Graph &IfOp::getThenGraph() const {
  return getGraph().getIr().getGraph(thenGraphId);
}

Graph &IfOp::getElseGraph() const {
  return getGraph().getIr().getGraph(elseGraphId);
}

std::vector<const Graph *> IfOp::getCalledGraphs() const {
  return {&getThenGraph(), &getElseGraph()};
}

InIndex IfOp::opInToSubgraphInIndex(SubgraphIndex subgraphIndex,
                                    InIndex inIndex) const {
  if (!input->hasIndex(inIndex)) {
    throw error(
        "Invalid inIndex for Op {} (op does not have an input with index {})",
        debugName(),
        inIndex);
  }

  if (subgraphIndex == thenSubgraphIndex) {
    if (thenInputIndicesMap.find(inIndex) != thenInputIndicesMap.end()) {
      return thenInputIndicesMap.at(inIndex);
    } else {
      return -1;
    }
  } else if (subgraphIndex == elseSubgraphIndex) {
    if (elseInputIndicesMap.find(inIndex) != elseInputIndicesMap.end()) {
      return elseInputIndicesMap.at(inIndex);
    } else {
      return -1;
    }
  } else {
    throw error("Invalid subgraphIndex for {} (expected 0 or 1, got {})",
                debugName(),
                subgraphIndex);
  }
}

InIndex IfOp::subgraphInToOpInIndex(SubgraphIndex subgraphIndex,
                                    InIndex inIndex) const {

  auto getInIndex = [inIndex](Graph &subgraph,
                              const std::map<InIndex, InIndex> &map) {
    if (inIndex < 0 || inIndex >= subgraph.getInputIds().size()) {
      throw error("Invalid inIndex for subgraph '{}' (subgraph does not have "
                  "an input with index {})",
                  subgraph.id.str(),
                  inIndex);
    }

    // NOTE: We currently don't have a reverse mapping pre-calculated. If
    // performance is an issue, precalculate this mapping akin to subgraphops.
    for (const auto &entry : map) {
      if (entry.second == inIndex) {
        return entry.first;
      }
    }
    return -1;
  };

  if (subgraphIndex == thenSubgraphIndex) {
    return getInIndex(getThenGraph(), thenInputIndicesMap);
  } else if (subgraphIndex == elseSubgraphIndex) {
    return getInIndex(getElseGraph(), elseInputIndicesMap);
  } else {
    throw error("Invalid subgraphIndex for {} (expected 0 or 1, got {})",
                debugName(),
                subgraphIndex);
  }
}

OutIndex IfOp::opOutToSubgraphOutIndex(SubgraphIndex subgraphIndex,
                                       OutIndex outIndex) const {
  if (!output->hasIndex(outIndex)) {
    throw error(
        "Invalid outIndex for Op {} (op does not have an output with index {})",
        debugName(),
        outIndex);
  }

  if (subgraphIndex == thenSubgraphIndex) {
    if (thenOutputIndicesMap.find(outIndex) != thenOutputIndicesMap.end()) {
      return thenOutputIndicesMap.at(outIndex);
    } else {
      return -1;
    }
  } else if (subgraphIndex == elseSubgraphIndex) {
    if (elseOutputIndicesMap.find(outIndex) != elseOutputIndicesMap.end()) {
      return elseOutputIndicesMap.at(outIndex);
    } else {
      return -1;
    }
  } else {
    throw error("Invalid subgraphIndex for {} (expected 0 or 1, got {})",
                debugName(),
                subgraphIndex);
  }
}

OutIndex IfOp::subgraphOutToOpOutIndex(SubgraphIndex subgraphIndex,
                                       OutIndex outIndex) const {

  auto getOutIndex = [outIndex](Graph &subgraph,
                                const std::map<OutIndex, OutIndex> &map) {
    if (outIndex < 0 || outIndex >= subgraph.getOutputIds().size()) {
      throw error("Invalid outIndex for subgraph '{}' (subgraph does not have "
                  "an output with index {})",
                  subgraph.id.str(),
                  outIndex);
    }

    // NOTE: We currently don't have a reverse mapping pre-calculated. If
    // performance is an issue, precalculate this mapping akin to subgraphops.
    for (const auto &entry : map) {
      if (entry.second == outIndex) {
        return entry.first;
      }
    }
    return -1;
  };

  if (subgraphIndex == thenSubgraphIndex) {
    return getOutIndex(getThenGraph(), thenOutputIndicesMap);
  } else if (subgraphIndex == elseSubgraphIndex) {
    return getOutIndex(getElseGraph(), elseOutputIndicesMap);
  } else {
    throw error("Invalid subgraphIndex for {} (expected 0 or 1, got {})",
                debugName(),
                subgraphIndex);
  }
}

std::set<OutIndex> IfOp::opInToOpOutIndex(InIndex in) const { return {}; }

std::set<InIndex> IfOp::opOutToOpInIndex(OutIndex out) const { return {}; }

float IfOp::calcAutoVirtualGraphCost(std::set<int> &inputs_seen) {
  return 0.0f;
}

void IfOp::setCalledSubgraphGradInfo(
    const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) {
  calledGraphGradOpHelper.setCalledSubgraphGradInfo(calledGraphsGradInfo);
}

IfGradOp::IfGradOp(const IfOp &fwdOp,
                   const std::vector<GradInOutMapper> &gradInInfo_,
                   const BranchInfo &thenBranchInfo,
                   const BranchInfo &elseBranchInfo)
    : IfOp(Onnx::CustomGradOperators::IfGrad,
           thenBranchInfo,
           elseBranchInfo,
           fwdOp.getSettings()),
      gradInInfo(gradInInfo_) {

  // An output for every input except the condition
  for (int i = 1; i < fwdOp.input->n(); i++) {
    outInfoMap.insert({i - 1, i});
  }
}

std::unique_ptr<Op> IfGradOp::clone() const {
  return std::make_unique<IfGradOp>(*this);
}

const std::vector<GradInOutMapper> &IfGradOp::gradInputInfo() const {
  return gradInInfo;
}

const std::map<int, int> &IfGradOp::gradOutToNonGradIn() const {
  return outInfoMap;
}

IfConditionGradOp::IfConditionGradOp(const IfOp &fwdOp)
    : IdentityOp(Onnx::CustomGradOperators::IfConditionGrad,
                 fwdOp.getSettings()) {}

std::unique_ptr<Op> IfConditionGradOp::clone() const {
  return std::make_unique<IfConditionGradOp>(*this);
}

const std::vector<GradInOutMapper> &IfConditionGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), IfOp::getConditionInIndex(), GradOpInType::In}};

  return inInfo;
}

const std::map<int, int> &IfConditionGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), IfOp::getConditionInIndex()}};
  return outInfo;
}

namespace {

static OpDefinition::DataTypes B = {DataType::BOOL};
static OpDefinition::DataTypes V = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOL};

static OpDefinition ifOpDef({OpDefinition::Inputs({{"cond", B}}),
                             OpDefinition::Outputs({{"outputs", V}}),
                             OpDefinition::Attributes({
                                 {"else_branch", {"*"}},
                                 {"then_branch", {"*"}},
                             })});

static OpCreator<IfOp> ifOpCreator(
    OpDefinitions({{Onnx::Operators::If_1, ifOpDef},
                   {Onnx::Operators::If_11, ifOpDef}}),
    [](const OpCreatorInfo &info, Graph &graph) -> Op * {
      auto thenBranch =
          info.attributes.getAttribute<Attributes::Graph>("then_branch");
      auto elseBranch =
          info.attributes.getAttribute<Attributes::Graph>("else_branch");

      auto &parentGraph                             = info.settings.graph.get();
      auto &ir                                      = parentGraph.getIr();
      auto &tensors                                 = parentGraph.getTensors();
      const VectorAndSet<TensorId> &constTensorsIds = tensors.getConstIds();
      std::map<TensorId, TensorInfo> inputInfos;

      const auto setupInputInfos = [&](const bool inserted,
                                       const TensorId &inputId) {
        if (inserted)
          inputInfos[inputId] =
              tensors.get(addScope(parentGraph, inputId))->info;
      };

      // Collect all inputs names.
      bool inserted = false;
      FlatSet<TensorId> thenInputIds;
      for (auto &input : thenBranch.input()) {
        std::tie(std::ignore, inserted) = thenInputIds.insert(input.name());
        setupInputInfos(inserted, input.name());
      }
      FlatSet<TensorId> elseInputIds;
      for (auto &input : elseBranch.input()) {
        std::tie(std::ignore, inserted) = elseInputIds.insert(input.name());
        setupInputInfos(inserted, input.name());
      }

      const auto insertImplicitTensor = [&](FlatSet<TensorId> &branchInputIds,
                                            const TensorId &inputId,
                                            FlatSet<Tensor *> &constTensors) {
        if (constTensorsIds.contains(inputId)) {
          // If the tensor is a const, we put it in a separate container to
          // handle it differently.
          auto *tensor = tensors.get(inputId);
          constTensors.insert(tensor);
        } else {
          std::tie(std::ignore, inserted) = branchInputIds.insert(inputId);
          setupInputInfos(inserted, inputId);
        }
      };

      // Implicit const tensors used in branches.
      FlatSet<Tensor *> thenImplicitConstTensors;
      FlatSet<Tensor *> elseImplicitConstTensors;

      // Gather all implicit tensors from 'then' and 'else' branches.
      auto thenImplicitTensorIds = onnxutil::getImplicitTensorIds(thenBranch);
      for (const auto &implicitTensorId : thenImplicitTensorIds) {
        insertImplicitTensor(
            thenInputIds, implicitTensorId, thenImplicitConstTensors);
      }
      auto elseImplicitTensorIds = onnxutil::getImplicitTensorIds(elseBranch);
      for (const auto &implicitTensorId : elseImplicitTensorIds) {
        insertImplicitTensor(
            elseInputIds, implicitTensorId, elseImplicitConstTensors);
      }

      // Collect all outputs IDs.
      std::vector<TensorId> thenOutputIds;
      thenOutputIds.reserve(thenBranch.output().size());
      for (const auto &output : thenBranch.output()) {
        thenOutputIds.push_back(output.name());
      }
      std::vector<TensorId> elseOutputIds;
      elseOutputIds.reserve(elseBranch.output().size());
      for (const auto &output : elseBranch.output()) {
        elseOutputIds.push_back(output.name());
      }

      // Prepare branches graphs IDs.
      GraphId thenGraphId{nameBranchGraph(thenBranch, ir, "then_block")};
      GraphId elseGraphId{nameBranchGraph(elseBranch, ir, "else_block")};

      // Gather all the inputIds from both branches graphs.
      FlatSet<TensorId> inputIds;
      inputIds.reserve(thenInputIds.size() + elseInputIds.size());
      inputIds.insert(thenInputIds.begin(), thenInputIds.end());
      inputIds.insert(elseInputIds.begin(), elseInputIds.end());

      // Create maps of the op outputs to branch outputs.
      // In ONNX spec, then and else branches must have identical outputs.
      std::map<int, int> thenAndElseOutputIndicesMap;
      for (size_t i = 0; i < thenOutputIds.size(); i++) {
        thenAndElseOutputIndicesMap.insert({i, i});
      }

      // Create maps of the op inputs to branch inputs.
      const auto createInputIndicesMap =
          [&](const FlatSet<TensorId> &branchInputs) {
            std::map<int, int> branchInputIndicesMap;
            size_t idx = 0;
            for (const auto &id : inputIds) {
              const auto found = branchInputs.find(id);
              if (found !=
                  branchInputs.end()) { // && !constTensorsIds.contains(id)) {
                const auto branchIdx =
                    std::distance(branchInputs.begin(), found);
                // +1 required because of IfOp condition input
                branchInputIndicesMap.insert({idx + 1, branchIdx});
              }
              idx++;
            }
            return branchInputIndicesMap;
          };

      // Create IfOp.
      auto thenInputIndicesMap = createInputIndicesMap(thenInputIds);
      auto elseInputIndicesMap = createInputIndicesMap(elseInputIds);
      Op *op                   = graph.createOp<IfOp>(
          info.opid,
          BranchInfo{
              thenGraphId, thenInputIndicesMap, thenAndElseOutputIndicesMap},
          BranchInfo{
              elseGraphId, elseInputIndicesMap, thenAndElseOutputIndicesMap},
          info.settings);

      // Connect IfOp inputs.
      for (const auto &id : info.getInputIds()) {
        op->connectInTensor(op->input->n(), id);
      }
      for (const auto &inputId : inputIds) {
        op->connectInTensor(op->input->n(), addScope(parentGraph, inputId));
      }

      const auto constructBranchGraph =
          [&](Graph &branchGraph,
              Attributes::Graph &branch,
              const FlatSet<TensorId> &inputIds,
              const std::vector<TensorId> &outputIds,
              const FlatSet<Tensor *> &implicitConstTensors) {
            // Add regular graph inputs.
            for (const auto &id : inputIds) {
              const auto scopedId = addScope(branchGraph, id);
              branchGraph.addInput(scopedId, inputInfos.at(id));
            }
            // Add initialization of constants.
            for (const auto *tensor : implicitConstTensors) {
              const auto tId =
                  addScope(branchGraph, removeScope(graph, tensor->id));
              branchGraph.addConstInit(
                  tId, tensor->info, tensor->tensorData()->data(), tId);
            }
            // Add all the ops to the graph.
            branchGraph.constructFromOnnxGraph(branch);

            // Add graph outputs.
            for (const auto &id : outputIds) {
              const auto scopedId = addScope(branchGraph, id);
              branchGraph.markAsOutput(scopedId);
            }
          };

      // Construct then graph.
      auto &thenGraph = ir.createGraph(thenGraphId);
      constructBranchGraph(thenGraph,
                           thenBranch,
                           thenInputIds,
                           thenOutputIds,
                           thenImplicitConstTensors);

      // Construct else graph.
      auto &elseGraph = ir.createGraph(elseGraphId);
      constructBranchGraph(elseGraph,
                           elseBranch,
                           elseInputIds,
                           elseOutputIds,
                           elseImplicitConstTensors);

      // Connect IfOp outputs.
      for (const auto &id : info.getOutputIds())
        op->createAndConnectOutTensor(op->output->n(), id);

      return op;
    },
    true);
} // namespace

} // namespace popart
