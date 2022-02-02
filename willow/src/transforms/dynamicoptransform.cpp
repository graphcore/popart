// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/add.hpp>
#include <popart/op/dynamic/dynamicadd.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/dynamic/dynamiczero.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/sum.hpp>
#include <popart/patterns/inplace.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/dynamicoptransform.hpp>

namespace popart {

using TensorContext = std::tuple<VGraphId, ExecutionPhase, PipelineStage>;

std::size_t DynamicOpTransform::id() {
  return typeid(DynamicOpTransform).hash_code();
}

namespace {
void transfer(Op *from, Op *to) {

  if (from->hasVirtualGraphId()) {
    to->setVirtualGraphId(from->getVirtualGraphId());
  }
  if (from->hasExecutionPhase()) {
    to->setExecutionPhase(from->getExecutionPhase());
  }
  if (from->hasPipelineStage()) {
    to->setPipelineStage(from->getPipelineStage());
  }
  if (from->hasBatchSerializedPhase()) {
    to->setBatchSerializedPhase(from->getBatchSerializedPhase());
  }

  to->settings.recomputeType  = from->settings.recomputeType;
  to->settings.tensorLocation = from->settings.tensorLocation;
  to->fromLoss                = from->fromLoss;
  to->toLoss                  = from->toLoss;

  // Non-ref copy of input map, because inputs are modified in the loop
  auto inputMap = from->input->indicesMap();
  for (auto &input : inputMap) {
    for (auto &index : input.second) {
      from->disconnectInTensor(index, input.first);
      to->connectInTensor(index, input.first->id);
    }
  }

  // Non-ref copy of input map, because inputs are modified in the loop
  auto outputMap = from->output->indicesMap();
  for (auto &output : outputMap) {
    for (auto &index : output.second) {
      from->disconnectOutTensor(output.first);
      to->connectOutTensor(index, output.first->id);
    }
  }

  to->getGraph().topoCons->transfer(from, to);
  to->getGraph().eraseOp(from->id);
}
} // namespace

bool DynamicOpTransform::apply(Graph &graph) const {
  logging::transform::debug("[DynamicOpTransform] Started.");

  AliasModel aliasModel;
  AliasModelGrower aliasModelGrower{aliasModel};
  aliasModelGrower.growFullGraph(graph, DataDependenciesOnly::Yes);

  auto &ir      = graph.getIr();
  auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::Yes);

  std::map<Op *, std::vector<Op *>, POpCmp> opsToChainMap;

  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op = schedule[i];

    if (DynamicSlicePadGradOp *oldOp =
            dynamic_cast<DynamicSlicePadGradOp *>(op)) {
      if (oldOp->isNotOverlapping()) {
        std::unique_ptr<DynamicUpdateInplaceOp> newOp =
            std::make_unique<DynamicUpdateInplaceOp>(
                Onnx::CustomOperators::DynamicUpdateInplace,
                oldOp->getAxes(),
                oldOp->getSizes(),
                oldOp->isNotOverlapping(),
                op->getSettings());
        op = newOp.get();
        graph.moveIntoGraph(std::move(newOp));
        transfer(oldOp, op);
        schedule[i] = op;
      } else {
        std::unique_ptr<DynamicAddInplaceOp> newOp =
            std::make_unique<DynamicAddInplaceOp>(
                Onnx::CustomOperators::DynamicAddInplace,
                oldOp->getAxes(),
                oldOp->getSizes(),
                oldOp->isNotOverlapping(),
                op->getSettings());
        op = newOp.get();
        graph.moveIntoGraph(std::move(newOp));
        transfer(oldOp, op);
        schedule[i] = op;
      }

      Tensor *tensor = op->output->tensor(DynamicBaseOp::getOutIndex());
      auto consumers = tensor->consumers.getOps();
      if (consumers.size() > 1) {
        throw error(
            "[DynamicOpTransform] exactly one consumer of PadGrad expected");
      }

      if (consumers.size() == 1) {
        // The ops created here need chaining to either a constant (zero) tensor
        // or an existing gradient sum. Add them to opsToChainMap to chain at
        // the end of this function.
        opsToChainMap[consumers.front()].push_back(op);
      }
    }

    if (DynamicUpdateUpdaterGradOp *oldOp =
            dynamic_cast<DynamicUpdateUpdaterGradOp *>(op)) {
      std::unique_ptr<DynamicSliceOp> newOp = std::make_unique<DynamicSliceOp>(
          Onnx::CustomOperators::DynamicSlice_1,
          oldOp->getAxes(),
          oldOp->getSizes(),
          oldOp->isNotOverlapping(),
          op->getSettings());
      op = newOp.get();
      graph.moveIntoGraph(std::move(newOp));
      transfer(oldOp, op);
      op->setup();
      schedule[i] = op;
    }

    if (DynamicUpdateToUpdateGradOp *oldOp =
            dynamic_cast<DynamicUpdateToUpdateGradOp *>(op)) {
      if (oldOp->isNotOverlapping()) {
        Tensor *input = oldOp->input->tensor(
            DynamicUpdateToUpdateGradOp::getUpdateInIndex());
        Tensor *output = oldOp->output->tensors().front();
        for (Op *consumer : output->consumers.getOps()) {
          auto indices = consumer->input->indices(output);
          consumer->disconnectInTensor(output);
          for (auto index : indices) {
            consumer->connectInTensor(index, input->id);
          }
        }
        oldOp->disconnectAllInputs();
        oldOp->disconnectAllOutputs();
        oldOp->getGraph().eraseOp(oldOp->id);
        op          = nullptr;
        schedule[i] = nullptr;
      } else {
        std::unique_ptr<DynamicZeroOp> newOp = std::make_unique<DynamicZeroOp>(
            Onnx::CustomOperators::DynamicZero_1,
            oldOp->getAxes(),
            oldOp->getSizes(),
            oldOp->isNotOverlapping(),
            op->getSettings());
        op = newOp.get();
        graph.moveIntoGraph(std::move(newOp));
        transfer(oldOp, op);
        op->setup();
        schedule[i] = op;
      }
    }

    if (DynamicZeroGradOp *oldOp = dynamic_cast<DynamicZeroGradOp *>(op)) {
      std::unique_ptr<DynamicZeroOp> newOp =
          std::make_unique<DynamicZeroOp>(Onnx::CustomOperators::DynamicZero_1,
                                          oldOp->getAxes(),
                                          oldOp->getSizes(),
                                          oldOp->isNotOverlapping(),
                                          op->getSettings());
      op = newOp.get();
      graph.moveIntoGraph(std::move(newOp));
      transfer(oldOp, op);
      op->setup();
      schedule[i] = op;
    }

    // Dynamic update tensor layout preference is Update <- In
    if (dynamic_cast<DynamicTernaryBaseOp *>(op)) {
      if (op->settings.inferTensorMappingToFrom.empty()) {
        op->settings.inferTensorMappingToFrom.insert(
            {DynamicTernaryBaseOp::getUpdateInIndex(),
             DynamicTernaryBaseOp::getInIndex()});
      }
    }
  }

  chainDynamicInplaceGradOps(ir, opsToChainMap, aliasModel);

  logging::transform::debug("[DynamicOpTransform] Done.");
  return true;
}

void DynamicOpTransform::chainDynamicInplaceGradOps(
    Ir &ir,
    const std::map<Op *, std::vector<Op *>, POpCmp> &opsToChainMap,
    AliasModel &aliasModel) const {
  // opsToChainMap is a map from consumers of DynamicUpdateInplaceOp/
  // DynamicAddInplaceOp to those ops. Each DynamicUpdate op adds at most
  // one consumer, so if the value is a vector of length >1, these would
  // represent separate inputs to.

  for (const auto &kv : opsToChainMap) {
    Op *consumerOp = kv.first;

    SumOp *maybeSumOp = dynamic_cast<SumOp *>(consumerOp);
    AddOp *maybeAddOp = dynamic_cast<AddOp *>(consumerOp);

    if (maybeSumOp != nullptr) {
      gradSumToGradChain(ir, maybeSumOp, kv.second, aliasModel);
    } else if (maybeAddOp != nullptr && kv.second.size() == 2) {
      // Treat an add op like a sum op. This succeeds as long as
      // SumOp::getOutIndex() == AddOp::getOutIndex().
      if (SumOp::getOutIndex() != AddOp::getOutIndex()) {
        throw error("SumOp::getOutIndex() != AddOp::getOutIndex()");
      }
      gradSumToGradChain(ir, maybeAddOp, kv.second, aliasModel);
    } else {
      // Check that there is only one op consumed as otherwise there should have
      // been a grad sum. Create a sum op to reuse the code which assumes a sum
      // op. It will be erased anyway.
      if (kv.second.size() > 1) {
        throw error("Unhandled gradient consumng op in dynamicoptransform.");
      }

      DynamicTernaryBaseInplaceOp *dynamicOp =
          dynamic_cast<DynamicTernaryBaseInplaceOp *>(kv.second.front());
      if (dynamicOp == nullptr) {
        throw error("The op is not a subclass of DynamicTernaryBaseInplaceOp.");
      }

      // Find the input index of the consumer op which is conencted to the
      // dynamic op
      auto outTensor        = dynamicOp->outTensor(dynamicOp->getOutIndex());
      InIndex inTensorIndex = -1;
      for (InIndex idx = 0; idx < consumerOp->inTensorCount(); idx++) {
        if (consumerOp->inId(idx) == outTensor->id) {
          inTensorIndex = idx;
          break;
        }
      }
      if (inTensorIndex == -1) {
        throw error("Failed to find dynamic output tensor as input to the "
                    "consumer op in dynamicoptransform.");
      }

      // Create the sum op and add it to the graph
      auto sumOpUP =
          std::make_unique<SumOp>(Onnx::Operators::Sum_8,
                                  Op::Settings(kv.second.front()->getGraph(),
                                               "DynamicOpTransform_tempSum"));
      auto sumOp = sumOpUP.get();
      ir.getMainGraph().moveIntoGraph(std::move(sumOpUP));

      // Reconnect the tensor from the dynamic op to the sum op
      consumerOp->disconnectInTensor(inTensorIndex);
      sumOp->connectInTensor(0, outTensor->id);

      // Connect the sum op to the consumer op
      TensorId sumOut = ir.createIntermediateTensorId(outTensor->id);
      sumOp->createAndConnectOutTensor(sumOp->getOutIndex(), sumOut);
      consumerOp->connectInTensor(inTensorIndex, sumOut);
      sumOp->setup();

      // Process it as a sum
      gradSumToGradChain(ir, sumOp, kv.second, aliasModel);
    }
  }
}

void DynamicOpTransform::gradSumToGradChain(Ir &ir,
                                            Op *sumOp,
                                            std::vector<Op *> dynamicOps,
                                            AliasModel &aliasModel) const {

  Tensor *gradTensor = sumOp->output->tensor(SumOp::getOutIndex());

  // Maintain the last op in the chain
  TensorId lastId = ir.createIntermediateTensorId(gradTensor->id);

  // The sum op must be kept if not all inputs are in the dynamicOps to be
  // processed. The gradient from these will be summed before being updated o r
  // added to by the chained ops.
  bool sumMustBeKept = sumOp->input->n() > dynamicOps.size();

  // In any case, disconnect the current output tensor.
  sumOp->disconnectOutTensor(gradTensor);

  if (sumMustBeKept) {
    // Use the existing gradietn sum and add to or update this a as part of the
    // chain.
    sumOp->createAndConnectOutTensor(SumOp::getOutIndex(), lastId);
  } else {
    // Create a tensor of zeros for the gradient chain, and add to or update
    // this.
    auto initOp = std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                           gradTensor->info,
                                           TensorType::ActGrad,
                                           InitType::Zero,
                                           dynamicOps.front()->getSettings());
    Op *init    = initOp.get();
    init->setName("GradInit_" + gradTensor->id);
    if (dynamicOps.front()->hasBatchSerializedPhase()) {
      init->setBatchSerializedPhase(-1);
    }
    init->toLoss   = PathToLoss::No;
    init->fromLoss = PathFromLoss::Yes;
    ir.getMainGraph().moveIntoGraph(std::move(initOp));
    init->createAndConnectOutTensor(InitOp::getOutIndex(), lastId);
    init->setup();
  }

  // Disconnect dynamic ops as inputs to the sum op.
  for (size_t i = 0; i < dynamicOps.size(); ++i) {
    Op *op         = dynamicOps.at(i);
    auto tensorMap = op->output->tensorMap();
    for (auto indexAndTensor : tensorMap) {
      if (sumOp->input->indicesMap().find(indexAndTensor.second) !=
          sumOp->input->indicesMap().end()) {
        sumOp->disconnectInTensor(indexAndTensor.second);
      }
    }
  }

  // If the sum must be kept, it will need to be reconfigured. Otherwise,
  // simply erase it.
  if (sumMustBeKept) {
    sumOp->inheritPlacementAttributes(true, aliasModel);
    sumOp->setup();
  } else {
    ir.getMainGraph().eraseOp(sumOp->id);
  }

  // Go thorough and add ops to the gradient chain
  for (size_t i = 0; i < dynamicOps.size(); ++i) {
    bool isLast = i == dynamicOps.size() - 1;
    Op *op      = dynamicOps.at(i);
    op->connectInTensor(DynamicTernaryBaseOp::getUpdateInIndex(), lastId);
    op->disconnectAllOutputs();

    TensorId outId;

    if (isLast) {
      // Set the final output of the chain to the original grad tensor.
      outId = gradTensor->id;
      op->connectOutTensor(DynamicTernaryBaseOp::getOutIndex(), outId);
    } else {
      // Create a new tensor to add to the chain.
      outId = ir.createIntermediateTensorId(gradTensor->id);
      op->createAndConnectOutTensor(DynamicTernaryBaseOp::getOutIndex(), outId);
    }

    if (i == 0) {
      op->settings.inferTensorMappingToFrom.insert(
          {DynamicTernaryBaseOp::getUpdateInIndex(),
           DynamicTernaryBaseOp::getInIndex()});
    }

    op->setup();

    logging::transform::trace(
        "[DynamicOpTransform] Chaining gradients {} -> {} ops {} -> {}",
        lastId,
        outId,
        ir.getTensor(lastId)->getProducer()->debugName(),
        ir.getTensor(outId)->getProducer()->debugName());

    // Update the end of the chain.
    lastId = outId;
  }
}

namespace {
// DynamicOpTransform
bool init = Transform::registerTransform(new DynamicOpTransform());
} // namespace

} // namespace popart
