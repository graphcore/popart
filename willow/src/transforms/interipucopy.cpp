// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>
#include <popart/alias/aliasmodel.hpp>
#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/transpose.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/transforms/interipucopy.hpp>
#include <popart/util.hpp>

#include "popart/basicoptionals.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/opdebuginfo.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorlocation.hpp"
#include "popart/transforms/transform.hpp"
#include "popart/vertex.hpp"

namespace popart {

using IpuNumber = int64_t;

class CopiedTensors {

  // record which tensors have been copied to which ipu's
  std::map<TensorId, std::vector<IpuNumber>> tensorMap;

public:
  CopiedTensors() = default;

  // Return true if the tensor has been copied to the ipu
  bool find(TensorId id, IpuNumber ipuNumber) {
    bool found = false;
    auto it    = tensorMap.find(id);
    if (it != tensorMap.end()) {
      auto it2 = std::find(it->second.begin(), it->second.end(), ipuNumber);
      if (it2 != it->second.end()) {
        found = true;
      }
    }

    return found;
  }

  // Record that tensor (id) has been copied to ipuNumber
  void add(TensorId id, IpuNumber ipuNumber) {
    auto it = tensorMap.find(id);
    if (it != tensorMap.end()) {
      it->second.push_back(ipuNumber);
    } else {
      tensorMap.insert(std::make_pair(id, std::vector<IpuNumber>({ipuNumber})));
    }
  }

  void clear() { tensorMap.clear(); }

  const std::map<TensorId, std::vector<IpuNumber>> &getTensorMap() {
    return tensorMap;
  }
};

namespace {

// Checks if a tensor is a "view" of a weight. Not technically correct, since
// we also check outplace operations, but this is necessary to avoid copies
// when e.g. matmul serialisation leads to outplace copies of weights.
// Could be avoided if weights were properly aliased by using inplace view
// changes from the start (model definition and IR transformations)
bool isWeightOrConstView(Tensor *t) {
  bool is;
  graphutils::traverse(
      {t},
      [&is](Tensor *t0) {
        is |= (t0->tensorType() == TensorType::Variable ||
               t0->tensorType() == TensorType::Const);
        return true;
      },
      [](Op *op, Tensor *t0, Tensor *t1) {
        return op->isConvertibleTo<ReshapeBaseOp>() ||
               op->isConvertibleTo<TransposeBaseOp>() ||
               op->isConvertibleTo<SliceOp>() ||
               op->isConvertibleTo<SliceInplaceOp>() ||
               op->isConvertibleTo<SliceGradOp>() ||
               op->isConvertibleTo<ConcatOp>() ||
               op->isConvertibleTo<ConcatInplaceOp>() ||
               op->isConvertibleTo<ConcatGradOp>();
      },
      graphutils::TraversalType::DepthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::Backward,
      graphutils::TraverseCallSites::Current);
  return is;
}

struct PipelineStageAndVGraphIdLtCmp {
  bool operator()(std::pair<PipelineStage, VGraphId> const &a,
                  std::pair<PipelineStage, VGraphId> const &b) const {
    if (a.first != unusedPipelineStage &&
        (a.first < b.first || b.first == unusedPipelineStage)) {
      return true;
    }
    if (b.first != unusedPipelineStage &&
        (b.first < a.first || a.first == unusedPipelineStage)) {
      return false;
    }
    if (a.second < b.second) {
      return true;
    } else {
      return false;
    }
  }
};

struct PipelineStageAndVGraphIdGtCmp {
  bool operator()(std::pair<PipelineStage, VGraphId> const &a,
                  std::pair<PipelineStage, VGraphId> const &b) const {
    if (a.first != unusedPipelineStage &&
        (a.first > b.first || b.first == unusedPipelineStage)) {
      return true;
    }
    if (b.first != unusedPipelineStage &&
        (b.first > a.first || a.first == unusedPipelineStage)) {
      return false;
    }
    if (a.second > b.second) {
      return true;
    } else {
      return false;
    }
  }
};

// This transform may happen when pipelining disabled, in which case it should
// fall back to vgraph id.
std::pair<PipelineStage, VGraphId>
getInPipelineStageAndVGraphId(const Op *op, InIndex index) {
  std::pair<PipelineStage, VGraphId> pipelineAndVgraphId;
  pipelineAndVgraphId.first =
      op->hasPipelineStage() ? op->getPipelineStage() : unusedPipelineStage;
  std::set<OpId> visited;
  pipelineAndVgraphId.second =
      op->getIntrospectionInVirtualGraphId(index, visited).first;
  return pipelineAndVgraphId;
}

bool belongsInOptimizerFromHostFragment(const Graph &graph, IpuCopyOp *copyOp) {
  if (copyOp->copiesOptimizerTensors()) {
    if (copyOp->inTensor(0)->hasProducer() &&
        copyOp->inTensor(0)->getProducer()->settings.executionContext !=
            ExecutionContext::OptimizerFromHostFragment) {
      return false;
    }

    else if (copyOp->getIr()
                 .getSessionOptions()
                 .automaticLossScalingSettings.enabled) {
      bool copiesLossScaleTensor =
          copyOp->inTensor(0) == getLossScaleTensor(graph);
      auto inverseLossScaleTensors = getInverseLossScaleTensors(graph);
      bool copiesInverseLossScaleTensor =
          inverseLossScaleTensors.find(copyOp->inTensor(0)) !=
          inverseLossScaleTensors.end();
      if (copiesLossScaleTensor || copiesInverseLossScaleTensor) {
        // If auto loss scaling is enabled, and the copy op copies the loss
        // scale tensor, or the inverse loss scale tensor, then it must
        // execute in the Normal execution context.
        return false;
      } else {
        return true;
      }
    } else {
      return true;
    }
  }

  // Non optimizer-tensor-copying op
  return false;
}

bool outputsConsumedByAccumulateOuterFragmentOps(const Op *copyOp) {
  auto consumers = copyOp->outTensor(0)->consumers.getOps();
  bool allConsumersInAccumulateOuterFragment = true;
  for (Op *consumer : consumers) {
    if (consumer->settings.executionContext !=
        ExecutionContext::AccumulateOuterFragment) {
      allConsumersInAccumulateOuterFragment = false;
    }
  }
  return allConsumersInAccumulateOuterFragment;
}

void setPlacementAttributes(CopiedTensors copiedTensors,
                            const Graph &graph,
                            AliasModel &aliasModel) {
  for (auto &copied : copiedTensors.getTensorMap()) {
    for (Op *op : graph.getTensors().get(copied.first)->consumers.getOps()) {
      if (op->isIpuCopyOp()) {
        auto copyOp = dynamic_cast<IpuCopyOp *>(op);
        copyOp->inheritPlacementAttributes(false, aliasModel);
      }
    }
  }
}

void setExecutionContext(CopiedTensors copiedTensors, const Graph &graph) {
  for (auto &copied : copiedTensors.getTensorMap()) {
    for (Op *op : graph.getTensors().get(copied.first)->consumers.getOps()) {
      if (op->isIpuCopyOp()) {
        auto copyOp = dynamic_cast<IpuCopyOp *>(op);

        // Set the execution context of the copy op in special case
        if (belongsInOptimizerFromHostFragment(graph, copyOp)) {
          copyOp->settings.executionContext =
              ExecutionContext::OptimizerFromHostFragment;
        } else if (outputsConsumedByAccumulateOuterFragmentOps(copyOp)) {
          copyOp->settings.executionContext =
              ExecutionContext::AccumulateOuterFragment;
        }
      }
    }
  }
}

} // namespace

std::size_t InterIpuCopy::id() { return typeid(InterIpuCopy).hash_code(); }

TensorId InterIpuCopy::generateCopiedTensorId(Tensor *tensor,
                                              VGraphId toIpu) const {
  // The copiedTensor id needs to be unique as the same tensor may be copied to
  // multiple ipus's
  TensorId copiedTensor = tensor->id + "_c" + std::to_string(toIpu);
  return copiedTensor;
}

void InterIpuCopy::connectIpuCopy(Graph &,
                                  Tensor *tensor,
                                  Op *fromOp,
                                  VGraphId fromIpu,
                                  Op *toOp,
                                  VGraphId toIpu) const {

  if (fromOp != toOp) {
    // We have already copied this tensor but we still need to
    // update the 'to' op to use the copied tensor
    logging::transform::debug("Already copied output tensor of {}:{} from "
                              "ipu {} to ipu {}",
                              fromOp->debugName(),
                              tensor->id,
                              fromIpu,
                              toIpu);
  } else {
    // For graph inputs
    logging::transform::debug(
        "Already copied output tensor of graph {} input {} from "
        "ipu {} to ipu {}",
        tensor->getGraph().id.str(),
        tensor->id,
        fromIpu,
        toIpu);
  }

  // Copy the list of index's this input tensor is mapped
  auto indices = toOp->input->indices(tensor);

  // Remove this input tensor from the to op for each index
  for (auto i : indices) {
    logging::transform::debug(
        "Disconnecting out {} from {}:{}", tensor->id, toOp->debugName(), i);
    toOp->disconnectInTensor(i, tensor);
  }

  // The copiedTensor id needs to be unique as the same tensor may be copied to
  // multiple ipus's
  TensorId copiedTensor = generateCopiedTensorId(tensor, toIpu);

  // Add the copied input tensor to the to op for each index
  for (auto i : indices) {
    logging::transform::debug(
        "Connecting in {} from {}:{}", copiedTensor, toOp->debugName(), i);
    toOp->connectInTensor(i, copiedTensor);
  }
}

void InterIpuCopy::insertIpuCopy(Graph &graph,
                                 Tensor *tensor,
                                 Op *fromOp,
                                 VGraphId fromIpu,
                                 Op *toOp,
                                 VGraphId toIpu) const {

  if (fromOp != toOp) {
    // Need to copy the tensor between ipu's
    logging::transform::debug(
        "Adding copy of output tensor of {}:{} from ipu {} to ipu {}",
        fromOp->debugName(),
        tensor->id,
        fromIpu,
        toIpu);
  } else {
    // For graph inputs
    logging::transform::debug("Adding copy of output tensor of graph {} input "
                              "{} from ipu {} to ipu {}",
                              tensor->getGraph().id.str(),
                              tensor->id,
                              fromIpu,
                              toIpu);
  }

  Op::Settings settings(graph, "");

  // Link debug information to fromOp
  settings.debugInfoId = fromOp->debugInfo.getId();

  // Inherit important settings from the fromOp
  // Tensor caching is inherited
  settings.tensorLocation = fromOp->getSettings().tensorLocation;

  auto ipuCopy_op = std::make_unique<IpuCopyOp>(
      Onnx::CustomOperators::IpuCopy, toIpu, settings);

  auto ipuCopy = ipuCopy_op.get();
  graph.moveIntoGraph(std::move(ipuCopy_op));

  // Copy the list of index's this input tensor is mapped
  auto indices = toOp->input->indices(tensor);

  // Remove this input tensor from the to op for each index
  for (auto i : indices) {
    logging::transform::debug(
        "Disconnecting in {} from {}:{}", tensor->id, toOp->debugName(), i);
    toOp->disconnectInTensor(i, tensor);
  }

  ipuCopy->connectInTensor(0, tensor->id, fromIpu);

  // The copiedTensor id needs to be unique as the same tensor may be copied to
  // multiple ipus's
  TensorId copiedTensor = generateCopiedTensorId(tensor, toIpu);

  ipuCopy->createAndConnectOutTensor(0, copiedTensor);
  ipuCopy->setup();

  // Add the copied input tensor to the to op for each index
  for (auto i : indices) {
    logging::transform::debug(
        "Connecting in {} to {}:{}", copiedTensor, toOp->debugName(), i);
    toOp->connectInTensor(i, copiedTensor);
  }
}

bool InterIpuCopy::apply(Graph &graph) const {
  // If the first op does not have an ipuNumber attribute, assume that no op's
  // have the ipuNumber set and so there is no inter ipu copy required.
  if (graph.getOps().size() > 0 &&
      !(graph.getOps().begin()->second->hasVirtualGraphId())) {
    return false;
  }

  AliasModel aliasModel;
  AliasModelGrower aliasModelGrower{aliasModel};
  aliasModelGrower.growFullGraph(graph, DataDependenciesOnly::Yes);

  // Keep a record of which tensors have been copied to which ipu's so we don't
  // duplicate a copy of a tensor between ipus
  CopiedTensors copiedTensors;

  // Keep a record of which stream tensors are going to which ops
  std::map<TensorId, std::set<OpId>> streamsMap;

  // For each graph input
  for (auto tid : graph.getInputIds()) {
    auto tensor             = graph.getTensors().get(tid);
    auto fromIpu            = tensor->getVirtualGraphIdUnsafe();
    auto fromPipelineStages = tensor->getPipelineStages();

    // For each consumer op of the tensor
    // but, take a copy of the map as we will be modifying it.
    std::map<Op *, int, POpCmp> map = tensor->consumers.getMap();
    for (auto &c : map) {

      Op *to          = c.first;
      InIndex toInIdx = to->input->indices(tensor).front();

      if (to->opid != Onnx::CustomOperators::IpuCopy) {

        // Get which ipu the tensor is supposed to be on
        std::set<OpId> visited;
        VGraphId toIpu =
            to->getIntrospectionInVirtualGraphId(toInIdx, visited).first;
        OptionalPipelineStage toPipelineStage = to->getOptionalPipelineStage();

        bool implicitPipelineFwdOnlyCopy =
            (graph.getIr()
                 .getSessionOptions()
                 .createImplicitPipeliningFwdOnlyProgram &&
             toPipelineStage && fromPipelineStages.size() &&
             (*toPipelineStage == (*fromPipelineStages.begin()) + 1) &&
             !isWeightOrConstView(tensor));

        // If the ops are not on the same ipu, or adjacent pipeline stages
        // with implicitPipelineFwdOnlyCopy
        if (fromIpu != toIpu || implicitPipelineFwdOnlyCopy) {

          bool alreadyCopied = copiedTensors.find(tensor->id, toIpu);

          if (alreadyCopied == true) {
            connectIpuCopy(graph, tensor, to, fromIpu, to, toIpu);
          } else {
            insertIpuCopy(graph, tensor, to, fromIpu, to, toIpu);

            // Record the copy
            copiedTensors.add(tensor->id, toIpu);
          }
        }
      }
    }
  }

  // For each op
  for (auto &entry : graph.getOps()) {

    Op *from = entry.second.get();

    if (from->opid != Onnx::CustomOperators::IpuCopy) {

      // For each input tensor
      auto &input = from->input;
      for (auto &t : input->tensorMap()) {

        Tensor *tensor = t.second;

        // Record the tensors so we can later work out if any input
        // tensor is going to two ipus
        if (tensor->tensorType() == TensorType::Stream ||
            tensor->tensorType() == TensorType::Const ||
            tensor->tensorType() == TensorType::Variable) {
          auto it = streamsMap.find(tensor->id);
          if (it == streamsMap.end()) {
            std::set<OpId> streams = {from->id};
            streamsMap.insert(std::make_pair(tensor->id, streams));
          } else {
            streamsMap[tensor->id].insert(from->id);
          }
        }
      }

      // For each output tensor
      auto &output = from->output;
      for (auto &t : output->tensorMap()) {

        Tensor *tensor = t.second;
        VGraphId fromIpu =
            from->getIntrospectionOutVirtualGraphId(t.first).first;
        OptionalPipelineStage fromPipelineStage =
            from->getOptionalPipelineStage();

        // For each consumer op of the tensor
        // but, take a copy of the map as we will be modifying it.
        std::map<Op *, int, POpCmp> map = t.second->consumers.getMap();
        for (auto &c : map) {

          Op *to          = c.first;
          InIndex toInIdx = to->input->indices(tensor).front();
          if (to->opid != Onnx::CustomOperators::IpuCopy) {

            // Get which ipu the tensor is supposed to be on
            VGraphId toIpu =
                to->getIntrospectionInVirtualGraphId(toInIdx).first;
            OptionalPipelineStage toPipelineStage =
                to->getOptionalPipelineStage();

            bool implicitPipelineFwdOnlyCopy =
                (graph.getIr()
                     .getSessionOptions()
                     .createImplicitPipeliningFwdOnlyProgram &&
                 (fromPipelineStage && toPipelineStage &&
                  *toPipelineStage == *fromPipelineStage + 1) &&
                 !isWeightOrConstView(tensor));

            // If the ops are not on the same ipu, or adjacent pipeline stages
            // with implicitPipelineFwdOnlyCopy
            if (fromIpu != toIpu || implicitPipelineFwdOnlyCopy) {

              bool alreadyCopied = copiedTensors.find(tensor->id, toIpu);

              if (alreadyCopied == true) {
                connectIpuCopy(graph, tensor, from, fromIpu, to, toIpu);
              } else {
                insertIpuCopy(graph, tensor, from, fromIpu, to, toIpu);

                // Record the copy
                copiedTensors.add(tensor->id, toIpu);
              }
            }
          }
        }
      }
    }
  }
  setExecutionContext(copiedTensors, graph);

  // For any stream tensors that are mapped to multiple ipus we will
  // use the first ipu the list as the input from the host and then
  // add ops in copy that tensor to other ipus
  copiedTensors.clear();
  for (auto &s : streamsMap) {
    auto &streamId    = s.first;
    auto &consumerIds = s.second;
    auto tensor       = graph.getTensors().get(s.first);

    if (consumerIds.size() > 1) {
      std::vector<Op *> consumers;
      for (auto id : consumerIds) {
        consumers.push_back(graph.getOp(id));
      }

      // Sort by smallest valid PipelineStage & VGraphId
      std::set<std::pair<PipelineStage, VGraphId>,
               PipelineStageAndVGraphIdLtCmp>
          sourceIpusLt;

      // Sort by largest valid PipelineStage & VGraphId
      std::set<std::pair<PipelineStage, VGraphId>,
               PipelineStageAndVGraphIdGtCmp>
          sourceIpusGt;

      for (Op *op : consumers) {
        auto consumerIpu = getInPipelineStageAndVGraphId(
            op, op->input->indices(tensor).front());
        sourceIpusLt.insert(consumerIpu);
        sourceIpusGt.insert(consumerIpu);
      }

      // sourceIpu should be the smallest VGraphId
      auto sourceIpu = *(sourceIpusLt.begin());

      if (graph.getIr().getSessionOptions().enablePipelining) {
        // We add an additional constraint on the order of IPU copies here:
        //
        // 'All stream tensors scheduled pre-loss are to be copied to a
        //  larger IPU number than that of sourceOp.
        //  All stream tensors scheduled post-loss are to be copied to a
        //  smaller IPU number than that of sourceOp'
        //
        if (tensor->scheduledPreLoss != ScheduledPreLoss::Yes) {
          // sourceOp should be op mapped to largest VGraphId
          sourceIpu = *(sourceIpusGt.begin());
        }
      }

      logging::transform::debug(
          "Mapping stream tensor {} to PipelineStage {} VirtualGraphId {}",
          tensor->id,
          sourceIpu.first,
          sourceIpu.second);

      for (auto &op : consumers) {
        VGraphId toIpu = op->getIntrospectionInVirtualGraphId(
                               op->input->indices(tensor).front())
                             .first;

        // It the case of the first op the ipu will be the same so nothing to do
        if (sourceIpu.second != toIpu) {

          logging::transform::debug(
              "Adding op to copy streaming tensor {} from ipu {} to ipu {}",
              streamId,
              sourceIpu.second,
              toIpu);

          Tensor *tensor = graph.getTensors().get(streamId);

          bool alreadyCopied = copiedTensors.find(tensor->id, toIpu);
          if (alreadyCopied == true) {
            connectIpuCopy(graph, tensor, op, sourceIpu.second, op, toIpu);
          } else {
            insertIpuCopy(graph, tensor, op, sourceIpu.second, op, toIpu);

            // Record the copy
            copiedTensors.add(tensor->id, toIpu);
          }
        }
      }
    }
  }
  setPlacementAttributes(copiedTensors, graph, aliasModel);
  setExecutionContext(copiedTensors, graph);

  return true;
}

namespace {
bool init = Transform::registerTransform(new InterIpuCopy);
}

} // namespace popart
