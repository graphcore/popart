// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/dynamic/dynamicadd.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/dynamic/dynamiczero.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/sum.hpp>
#include <popart/patterns/inplace.hpp>
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

  auto &ir      = graph.getIr();
  auto schedule = graph.getOpSchedule({});

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

  gradSumToGradChain(ir, opsToChainMap);

  logging::transform::debug("[DynamicOpTransform] Done.");
  return true;
}

void DynamicOpTransform::gradSumToGradChain(
    Ir &ir,
    std::map<Op *, std::vector<Op *>, POpCmp> opsToChainMap) const {

  TensorId lastId;
  for (auto kv : opsToChainMap) {
    Op *sumOp          = kv.first;
    Tensor *gradTensor = sumOp->output->tensor(SumOp::getOutIndex());
    bool sumRequired   = sumOp->input->n() > kv.second.size();
    lastId             = ir.createIntermediateTensorId(gradTensor->id);
    sumOp->disconnectOutTensor(gradTensor);

    if (!sumRequired) {
      // No gradient to start the chain
      auto initOp = std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                             gradTensor->info,
                                             TensorType::ActGrad,
                                             InitType::Zero,
                                             kv.second.front()->getSettings());
      Op *init    = initOp.get();
      init->setName("GradInit_" + gradTensor->id);
      if (kv.second.front()->hasBatchSerializedPhase()) {
        init->setBatchSerializedPhase(-1);
      }
      init->toLoss   = PathToLoss::No;
      init->fromLoss = PathFromLoss::Yes;
      ir.getMainGraph().moveIntoGraph(std::move(initOp));
      init->createAndConnectOutTensor(InitOp::getOutIndex(), lastId);
      init->setup();
    } else {
      // Gradient tensor to start chain from exists already
      sumOp->createAndConnectOutTensor(SumOp::getOutIndex(), lastId);
    }

    for (size_t i = 0; i < kv.second.size(); ++i) {
      Op *op         = kv.second.at(i);
      auto tensorMap = op->output->tensorMap();
      for (auto indexAndTensor : tensorMap) {
        if (sumOp->input->indicesMap().find(indexAndTensor.second) !=
            sumOp->input->indicesMap().end()) {
          sumOp->disconnectInTensor(indexAndTensor.second);
        }
      }
    }

    if (sumRequired) {
      sumOp->inheritPlacementAttributes(true);
      sumOp->setup();
    } else {
      ir.getMainGraph().eraseOp(sumOp->id);
    }

    for (size_t i = 0; i < kv.second.size(); ++i) {
      bool isLast = i == kv.second.size() - 1;
      Op *op      = kv.second.at(i);
      op->connectInTensor(DynamicTernaryBaseOp::getUpdateInIndex(), lastId);
      op->disconnectAllOutputs();

      TensorId outId;

      if (isLast) {
        outId = gradTensor->id;
        op->connectOutTensor(DynamicTernaryBaseOp::getOutIndex(), outId);
      } else {
        outId = ir.createIntermediateTensorId(gradTensor->id);
        op->createAndConnectOutTensor(DynamicTernaryBaseOp::getOutIndex(),
                                      outId);
      }

      if (i == 0)
        op->settings.inferTensorMappingToFrom.insert(
            {DynamicTernaryBaseOp::getUpdateInIndex(),
             DynamicTernaryBaseOp::getInIndex()});

      op->setup();

      logging::transform::trace(
          "[DynamicOpTransform] Chaining gradients {} -> {} ops {} -> {}",
          lastId,
          outId,
          ir.getTensor(lastId)->getProducer()->debugName(),
          ir.getTensor(outId)->getProducer()->debugName());
      lastId = outId;
    }
  }
}

namespace {
// DynamicOpTransform
bool init = Transform::registerTransform(new DynamicOpTransform());
} // namespace

} // namespace popart
