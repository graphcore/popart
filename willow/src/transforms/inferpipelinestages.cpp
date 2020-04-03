// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/loss.hpp>
#include <popart/transforms/inferpipelinestages.hpp>

namespace popart {

std::size_t InferPipelineStages::id() {
  return typeid(InferPipelineStages).hash_code();
}

namespace {

bool isTransformRequired(Graph &graph) {
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (op->hasPipelineStage()) {
      // Pipeline stage has been set already.
      return false;
    }
  }
  // Pipeline stages have not been set.
  return true;
}

void verifyPipelineStagesAreUnset(Graph &graph) {
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (op->hasPipelineStage()) {
      throw error("{} already has the pipeline stage attribute set",
                  op->debugName());
    }
  }
}

void checkOrder(Op *producer, Op *consumer, Tensor *tensor) {
  if (producer->getPipelineStage() > consumer->getPipelineStage()) {
    std::stringstream ss;
    ss << logging::format(
        "Tensor {} is consumed in an earlier pipeline stage than "
        "it is produced",
        tensor->str());
    ss << logging::format("\nProducer {} has pipeline stage {}",
                          producer->debugName(),
                          producer->getPipelineStage());
    ss << logging::format("\nConsumer {} has pipeline stage {}",
                          consumer->debugName(),
                          consumer->getPipelineStage());
    throw error(ss.str());
  }
}

void validateTransform(Graph &graph) {
  for (auto tid : graph.getTensors().getAllTensorIds()) {
    auto t = graph.getTensors().get(tid);
    if (t->hasProducer()) {
      for (auto consumer : t->consumers.getOps()) {
        checkOrder(t->getProducer(), consumer, t);
      }
    }
  }
}

} // namespace

bool InferPipelineStages::apply(Graph &graph) const {
  if (!isTransformRequired(graph)) {
    return false;
  }

  verifyPipelineStagesAreUnset(graph);

  std::stringstream ss;
  ss << "Infering op pipeline stages through virtual graph ids:";
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    ss << logging::format(
        "\n  {}: {}", op->debugName(), op->getVirtualGraphId());
    op->setPipelineStage(op->getVirtualGraphId());
  }
  for (auto &loss : graph.getLosses()) {
    ss << logging::format(
        "\n  Loss({}): {}", loss->input(0), loss->getVirtualGraphId());
    loss->pipelineStage(loss->getVirtualGraphId());
  }
  logging::debug("{}", ss.str());

  validateTransform(graph);

  return true;
}

namespace {
bool init = Transform::registerTransform(new InferPipelineStages);
}

} // namespace popart
