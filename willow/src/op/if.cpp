#include <boost/range/algorithm/copy.hpp>

#include <memory>
#include <onnx/onnx_pb.h>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/if.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

namespace popart {

namespace {

GraphId generateSubgraphUniqueId(const std::string postfix) {
  static int uid = 0;
  return GraphId(logging::format("ifop_subgraph_{}_{}", uid++, postfix));
}

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
           const std::vector<TensorId> inputIds_,
           const BranchInfo &thenBranchInfo,
           const BranchInfo &elseBranchInfo,
           const Op::Settings &settings_)
    : Op(opid_, settings_), inputIds(inputIds_),
      thenInputIndicesMap(thenBranchInfo.inputIndicesMap),
      elseInputIndicesMap(elseBranchInfo.inputIndicesMap),
      thenOutputIndicesMap(thenBranchInfo.outputIndicesMap),
      elseOutputIndicesMap(elseBranchInfo.outputIndicesMap),
      thenGraphId(thenBranchInfo.graphId), elseGraphId(elseBranchInfo.graphId) {
}

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

  auto addInputTensors = [&](const Graph &fwdGraph, const Graph &gradGraph) {
    for (auto m : gradGraph.gradInputInfo()) {
      switch (m.type) {
      case GradOpInType::IN: {
        auto scopedId   = fwdGraph.getInputId(m.iNonGrad);
        auto unscopedId = fwdGraph.removeScope(scopedId);
        requiredGradOpInputs.insert(unscopedId);
        break;
      }
      case GradOpInType::GRADOUT: {
        auto scopedId = getThenGraph().getOutputId(m.iNonGrad);
        auto opOutId  = branchOutIdToOpOutId.at(scopedId);
        auto gradId   = getGradId(opOutId);
        requiredGradOpInputs.insert(gradId);
        break;
      }
      case GradOpInType::OUT:
      default:
        throw error("Unsupported GradOpInType {}", m.type);
      }
    }
  };

  addInputTensors(getThenGraph(), gradThenGraph);
  addInputTensors(getElseGraph(), gradElseGraph);

  // condition tensor must be first
  std::vector<TensorId> result{inId(getConditionInIndex())};
  copy(requiredGradOpInputs, std::back_inserter(result));
  return result;
}

std::map<TensorId, int>
IfOp::getOpInIdToBwdGraphInIndexMap(const Graph &fwdGraph,
                                    const Graph &bwdGraph) const {
  auto branchOutIdToOpOutId = getBranchOutIdToOpOutIdMap();

  std::map<TensorId, int> result;
  for (auto m : bwdGraph.gradInputInfo()) {
    switch (m.type) {
    case GradOpInType::IN: {
      // branch input to tensor id
      auto branchInId = fwdGraph.getInputId(m.iNonGrad);
      auto opInId     = fwdGraph.removeScope(branchInId);
      result.insert({opInId, m.iGrad});
      break;
    }
    case GradOpInType::GRADOUT: {
      auto branchOutId = fwdGraph.getOutputId(m.iNonGrad);
      auto opOutId     = branchOutIdToOpOutId.at(branchOutId);
      auto gradId      = getGradId(opOutId);
      result.insert({gradId, m.iGrad});
      break;
    }
    case GradOpInType::OUT:
    default:
      throw error("Unsupported GradOpInType {}", m.type);
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
        gradInInfo.push_back({gradOpInIdx, idx, GradOpInType::IN});
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
        gradInInfo.push_back({gradOpInIdx, idx, GradOpInType::GRADOUT});
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
    auto id = gradOpInputIds.at(i);
    gradOpInIdToGradOpInIdx.insert({id, i});
  }

  // get a map of IfGradOp input ids to bwdGraph input indices
  auto gradOpInIdToBwdGraphInIdx =
      getOpInIdToBwdGraphInIndexMap(fwdGraph, bwdGraph);

  auto bwdInputIndicesMap = getInIndicesMapForGradOp(gradOpInIdToGradOpInIdx,
                                                     gradOpInIdToBwdGraphInIdx);

  auto fwdInputIndicesMap  = getBranchInIndicesMap(fwdGraph);
  auto bwdOutputIndicesMap = getOutIndicesMapForGradOp(fwdInputIndicesMap);

  return {bwdGraph.id, bwdInputIndicesMap, bwdOutputIndicesMap};
};

std::vector<std::unique_ptr<Op>> IfOp::getGradOps() {
  auto bwdThenGraphId = generateSubgraphUniqueId("thenGrad");
  auto bwdElseGraphId = generateSubgraphUniqueId("elseGrad");

  auto &bwdThenGraph = getThenGraph().getBackwardsGraph(bwdThenGraphId);
  auto &bwdElseGraph = getElseGraph().getBackwardsGraph(bwdElseGraphId);

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

void IfOp::setup() {
  for (auto &inputId : inputIds) {
    connectInTensor(input->n(), inputId);
  }

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

std::vector<TensorId> IfOp::getInputsForGraph(const Graph &graph) const {
  auto &idxMap = getBranchInIndicesMap(graph);
  std::vector<TensorId> ids;
  for (int i = 0; i < input->n(); i++) {
    auto hasIndex = idxMap.find(i) != idxMap.end();
    if (hasIndex) {
      ids.push_back(inId(i));
    }
  }
  return ids;
}

IfGradOp::IfGradOp(const IfOp &fwdOp,
                   const std::vector<GradInOutMapper> &gradInInfo_,
                   const BranchInfo &thenBranchInfo,
                   const BranchInfo &elseBranchInfo)
    : IfOp(Onnx::CustomGradOperators::IfGrad,
           {},
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

const std::vector<GradInOutMapper> &IfConditionGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), IfOp::getConditionInIndex(), GradOpInType::IN}};

  return inInfo;
}

const std::map<int, int> &IfConditionGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), IfOp::getConditionInIndex()}};
  return outInfo;
}

namespace {

static OpCreator<IfOp> ifOpCreator(
    {Onnx::Operators::If_1, Onnx::Operators::If_11},
    [](const OperatorIdentifier &opid_,
       const Op::Settings &settings_,
       const Attributes &attr) -> std::unique_ptr<Op> {
      auto elseBranch = attr.getAttribute<Attributes::Graph>("else_branch");
      auto thenBranch = attr.getAttribute<Attributes::Graph>("then_branch");

      if (elseBranch.output().size() != thenBranch.output().size()) {
        throw error("IfOp: else_branch and then_branch have different outputs");
      }

      auto &tensors = settings_.graph.get().getTensors();
      std::map<TensorId, TensorInfo> inputInfos;

      // Collect all input names
      std::vector<TensorId> thenInputIds;
      for (auto &input : thenBranch.input()) {
        thenInputIds.push_back(input.name());
        inputInfos[input.name()] = tensors.get(input.name())->info;
      }
      std::vector<TensorId> elseInputIds;
      for (auto &input : elseBranch.input()) {
        elseInputIds.push_back(input.name());
        inputInfos[input.name()] = tensors.get(input.name())->info;
      }

      // Collect all output names
      std::vector<TensorId> thenOutputIds;
      for (auto &output : thenBranch.output()) {
        thenOutputIds.push_back(output.name());
      }
      std::vector<TensorId> elseOutputIds;
      for (auto &output : elseBranch.output()) {
        elseOutputIds.push_back(output.name());
      }

      auto thenGraphId = generateSubgraphUniqueId("then");
      auto elseGraphId = generateSubgraphUniqueId("else");

      // Construct then graph
      auto &ir        = settings_.graph.get().getIr();
      auto &thenGraph = ir.createGraph(thenGraphId);
      for (auto id : thenInputIds) {
        auto scopedId = thenGraph.addScope(id);
        thenGraph.addInput(scopedId, inputInfos.at(id));
      }
      thenGraph.constructFromOnnxGraph(thenBranch);
      for (auto id : thenOutputIds) {
        auto scopedId = thenGraph.addScope(id);
        thenGraph.markAsOutput(scopedId);
      }

      // Construct else graph
      auto &elseGraph = ir.createGraph(elseGraphId);
      for (auto id : elseInputIds) {
        auto scopedId = elseGraph.addScope(id);
        elseGraph.addInput(scopedId, inputInfos.at(id));
      }
      elseGraph.constructFromOnnxGraph(elseBranch);
      for (auto id : elseOutputIds) {
        auto scopedId = elseGraph.addScope(id);
        elseGraph.markAsOutput(scopedId);
      }

      // Get all the inputIds
      std::vector<TensorId> inputIds = [&]() {
        std::set<TensorId> inIds;
        inIds.insert(thenInputIds.begin(), thenInputIds.end());
        inIds.insert(elseInputIds.begin(), elseInputIds.end());
        return std::vector<TensorId>(inIds.begin(), inIds.end());
      }();

      // Create maps of the op inputs to branch inputs.
      auto createMap = [&inputIds](const std::vector<TensorId> &branchInputs) {
        std::map<int, int> branchInputIndicesMap;
        for (int i = 0; i < inputIds.size(); i++) {
          auto id    = inputIds.at(i);
          auto found = std::find(branchInputs.begin(), branchInputs.end(), id);
          if (found != branchInputs.end()) {
            auto branchIdx = std::distance(branchInputs.begin(), found);
            // +1 required because of IfOp condition input
            branchInputIndicesMap.insert({i + 1, branchIdx});
          }
        }
        return branchInputIndicesMap;
      };

      auto thenInputIndicesMap = createMap(thenInputIds);
      auto elseInputIndicesMap = createMap(elseInputIds);

      // Create maps of the op outputs to branch outputs.
      // In ONNX spec, then and else branches must have identical outputs.
      std::map<int, int> thenAndElseOutputIndicesMap;
      for (int i = 0; i < thenOutputIds.size(); i++) {
        thenAndElseOutputIndicesMap.insert({i, i});
      }

      auto op = std::make_unique<IfOp>(
          opid_,
          inputIds,
          BranchInfo{
              thenGraphId, thenInputIndicesMap, thenAndElseOutputIndicesMap},
          BranchInfo{
              elseGraphId, elseInputIndicesMap, thenAndElseOutputIndicesMap},
          settings_);

      return std::move(op);
    },
    true);
} // namespace

} // namespace popart
