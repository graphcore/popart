// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <vector>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/div.hpp>
#include <popart/op/max.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/reducesumsquare.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/sqrt.hpp>
#include <popart/op/sum.hpp>
#include <popart/opidentifier.hpp>
#include <popart/transforms/clipweightgradientsbynorm.hpp>

namespace popart {

namespace {

// Helper method for creating new ops.
// Taken from Pattern::transferBaseProperties.
void transferBaseProperties(Op *from, Op *to) {
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

  to->settings.scope            = from->settings.scope;
  to->settings.recomputeType    = from->settings.recomputeType;
  to->settings.tensorLocation   = from->settings.tensorLocation;
  to->fromLoss                  = from->fromLoss;
  to->toLoss                    = from->toLoss;
  to->settings.schedulePriority = from->settings.schedulePriority;
}

Tensor *addReduceSumSquare(Op *varUpdate, Graph &graph) {
  Op::Settings settings(graph, "", {});

  nonstd::optional<std::vector<int64_t>> axes;
  Op *reduction = graph.createOp<ReduceSumSquareOp>(
      Onnx::AiOnnx::OpSet11::ReduceSumSquare, axes, false, settings);

  transferBaseProperties(varUpdate, reduction);

  auto grad = varUpdate->inTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());

  auto &ir = graph.getIr();

  reduction->connectInTensor(ReduceSumSquareOp::getInIndex(), grad->id);
  reduction->createAndConnectOutTensor(ReduceSumSquareOp::getOutIndex(),
                                       ir.createIntermediateTensorId(grad->id));
  reduction->setup();
  return reduction->outTensor(ReduceSumSquareOp::getOutIndex());
}

// Return a consuming var update op for each tensor in weightIds.
std::vector<Op *> getVarUpdates(Graph &graph,
                                const std::vector<TensorId> &weightIds) {
  auto getVarUpdate = [](Tensor *t) {
    for (auto op : t->consumers.getOps()) {
      if (op->isConvertibleTo<SGD0VarUpdateOp>()) {
        return op;
      }
    }
    throw error("Could not find a varupdate op for tensor {}", t->id);
  };

  std::vector<Op *> varUpdates;
  for (auto &tid : weightIds) {
    auto tensor    = graph.getTensors().get(tid);
    auto varUpdate = getVarUpdate(tensor);
    varUpdates.push_back(varUpdate);
  }

  return varUpdates;
}

// globalNorm = sqrt(sum(gradNorms))
Tensor *createGlobalNorm(std::vector<Tensor *> gradNorms, Graph &graph) {
  Op::Settings settings(graph, "", {});

  Op *sum = graph.createOp<SumOp>(Onnx::AiOnnx::OpSet8::Sum, settings);
  transferBaseProperties(gradNorms.at(0)->getProducer(), sum);

  for (int i = 0; i < gradNorms.size(); i++) {
    auto gradNorm = gradNorms.at(i);
    sum->connectInTensor(i, gradNorm->id);
  }

  auto &ir = graph.getIr();
  sum->createAndConnectOutTensor(SumOp::getOutIndex(),
                                 ir.createIntermediateTensorId("normsSum"));
  sum->setup();

  Op *sqrt = graph.createOp<SqrtOp>(Onnx::AiOnnx::OpSet6::Sqrt, settings);
  transferBaseProperties(gradNorms.at(0)->getProducer(), sqrt);

  sqrt->connectInTensor(SqrtOp::getInIndex(), sum->outId(SumOp::getOutIndex()));
  sqrt->createAndConnectOutTensor(SqrtOp::getOutIndex(),
                                  ir.createIntermediateTensorId("globalNorm"));
  sqrt->setup();

  return sqrt->outTensor(SqrtOp::getOutIndex());
}

void addClipByNorm(Op *varUpdate, Tensor *clipFactor, Graph &graph) {
  auto &ir  = graph.getIr();
  auto grad = varUpdate->inTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());

  Op::Settings settings(graph, "", {});

  Op *mulOp = graph.createOp<MulOp>(Onnx::AiOnnx::OpSet6::Mul, settings);
  transferBaseProperties(varUpdate, mulOp);

  mulOp->connectInTensor(MulOp::getArg0InIndex(), grad->id);
  mulOp->connectInTensor(MulOp::getArg1InIndex(), clipFactor->id);
  mulOp->createAndConnectOutTensor(MulOp::getOutIndex(),
                                   ir.createIntermediateTensorId(grad->id));
  mulOp->setup();
  varUpdate->disconnectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                                grad);
  varUpdate->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                             mulOp->outId(MulOp::getOutIndex()));
}

void addClipByNorms(std::vector<Op *> varUpdates,
                    Tensor *clipFactor,
                    Graph &graph) {
  for (auto varUpdate : varUpdates) {
    addClipByNorm(varUpdate, clipFactor, graph);
  }
}

// clip_factor = clip_norm / max(clip_norm, global_norm)
Tensor *createClipFactor(Tensor *globalNorm, Tensor *clipNorm, Graph &graph) {
  auto &ir = graph.getIr();

  Op::Settings settings(graph, "", {});

  Op *maxOp = graph.createOp<MaxOp>(Onnx::AiOnnx::OpSet6::Max, settings);
  transferBaseProperties(globalNorm->getProducer(), maxOp);
  maxOp->connectInTensor(0, globalNorm->id);
  maxOp->connectInTensor(1, clipNorm->id);
  maxOp->createAndConnectOutTensor(MaxOp::getOutIndex(),
                                   ir.createIntermediateTensorId("clipByNorm"));
  maxOp->setup();

  Op *divOp = graph.createOp<DivOp>(Onnx::AiOnnx::OpSet6::Div, settings);
  transferBaseProperties(maxOp, divOp);
  divOp->connectInTensor(DivOp::getArg0InIndex(), clipNorm->id);
  divOp->connectInTensor(DivOp::getArg1InIndex(),
                         maxOp->outId(MaxOp::getOutIndex()));
  divOp->createAndConnectOutTensor(DivOp::getOutIndex(),
                                   ir.createIntermediateTensorId("clipByNorm"));
  divOp->setup();

  return divOp->outTensor(DivOp::getOutIndex());
}

// Take the maxNorm and create the clipNorm tensor.
Tensor *createClipNorm(float maxNorm, Graph &graph) {
  auto &ir          = graph.getIr();
  auto clipByNormId = ir.createIntermediateTensorId("clipByNorm");
  TensorInfo info{DataType::FLOAT, {}};
  std::vector<float> data{maxNorm};

  graph.getTensors().addConstInit(clipByNormId, info, data.data());
  return graph.getTensors().get(clipByNormId);
}

void clipWeightGradientsByNorm(const std::vector<TensorId> &weightIds,
                               float maxNorm,
                               Graph &graph) {
  auto varUpdates = getVarUpdates(graph, weightIds);

  std::vector<Tensor *> gradNorms;
  for (auto varUpdate : varUpdates) {
    auto gradNorm = addReduceSumSquare(varUpdate, graph);
    gradNorms.push_back(gradNorm);
  }

  auto globalNorm = createGlobalNorm(gradNorms, graph);
  auto clipNorm   = createClipNorm(maxNorm, graph);
  auto clipFactor = createClipFactor(globalNorm, clipNorm, graph);
  addClipByNorms(varUpdates, clipFactor, graph);
}

} // namespace

std::size_t ClipWeightGradientsByNorm::id() {
  return typeid(ClipWeightGradientsByNorm).hash_code();
}

bool ClipWeightGradientsByNorm::apply(Graph &graph) const {
  auto &ir        = graph.getIr();
  auto &optimizer = ir.getOptimizer();

  for (auto &clipGroup : optimizer.getClipNormSettings()) {
    clipWeightGradientsByNorm(clipGroup.weightIds, clipGroup.maxNorm, graph);
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new ClipWeightGradientsByNorm);
} // namespace

} // namespace popart
