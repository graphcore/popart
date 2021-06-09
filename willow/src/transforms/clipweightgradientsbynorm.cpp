// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <vector>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/adamupdater.hpp>
#include <popart/op/adamvarupdate.hpp>
#include <popart/op/div.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/max.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/reducesumsquare.hpp>
#include <popart/op/scale.hpp>
#include <popart/op/scaledadd.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <popart/op/sqrt.hpp>
#include <popart/op/sum.hpp>
#include <popart/opidentifier.hpp>
#include <popart/transforms/clipweightgradientsbynorm.hpp>
#include <popart/util.hpp>

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

ExecutionContext decideExecutionContext(Graph &graph) {
  auto &ir   = graph.getIr();
  auto &opts = ir.getSessionOptions();

  if (opts.enablePipelining && opts.enableGradientAccumulation) {
    return ExecutionContext::AccumulateOuterFragment;
  } else {
    return ExecutionContext::Normal;
  }
}

Tensor *addReduceSumSquare(Tensor *grad, Graph &graph) {
  logging::debug("addReduceSumSuareOp({}, graph)", grad->id);

  Op::Settings settings(graph, "", {});
  settings.executionContext = decideExecutionContext(graph);

  nonstd::optional<std::vector<int64_t>> axes;
  Op *reduction = graph.createOp<ReduceSumSquareOp>(
      Onnx::AiOnnx::OpSet11::ReduceSumSquare, axes, false, settings);

  transferBaseProperties(grad->consumers.getOps().at(0), reduction);

  auto &ir = graph.getIr();

  auto clippedGradId =
      logging::format("{}_{}", getBaseTensorId(grad->id), "clipping");

  reduction->connectInTensor(ReduceSumSquareOp::getInIndex(), grad->id);
  reduction->createAndConnectOutTensor(
      ReduceSumSquareOp::getOutIndex(),
      ir.createIntermediateTensorId(clippedGradId));
  reduction->setup();
  return reduction->outTensor(ReduceSumSquareOp::getOutIndex());
}

// Return a consuming var update op for each tensor in weightIds.
std::vector<Op *> getVarUpdates(Graph &graph,
                                const std::vector<TensorId> &weightIds) {
  auto getVarUpdate = [](Tensor *t) {
    logging::debug("Getting var updates for {}", t->id);
    for (auto op : t->consumers.getOps()) {
      if (op->isConvertibleTo<SGD0VarUpdateOp>() ||
          op->isConvertibleTo<SGD1VarUpdateOp>() ||
          op->isConvertibleTo<AdamVarUpdateOp>()) {
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

Tensor *createCopyOnVGraph(Tensor *t, VGraphId destination, Graph &graph) {
  auto &ir = graph.getIr();

  Op::Settings settings(graph, "", {});
  settings.executionContext = decideExecutionContext(graph);

  Op *op = graph.createOp<IpuCopyOp>(
      Onnx::CustomOperators::IpuCopy, destination, settings);
  IpuCopyOp *ipuCopy = dynamic_cast<IpuCopyOp *>(op);
  transferBaseProperties(t->getProducer(), op);

  ipuCopy->connectInTensor(0, t->id, t->getVirtualGraphId());
  ipuCopy->createAndConnectOutTensor(0, ir.createIntermediateTensorId(t->id));
  ipuCopy->setup();

  return ipuCopy->outTensor(0);
}

std::vector<Tensor *> copyToSameVGraph(const std::vector<Tensor *> ts,
                                       VGraphId destination,
                                       Graph &graph) {
  auto vGraphId = 0;
  std::vector<Tensor *> result;

  for (auto t : ts) {
    if (t->getVirtualGraphId() == destination) {
      result.push_back(t);
    } else {
      auto x = createCopyOnVGraph(t, vGraphId, graph);
      result.push_back(x);
    }
  }

  return result;
}

// globalNorm = sqrt(sum(gradNorms))
Tensor *createGlobalNorm(std::vector<Tensor *> gradNorms, Graph &graph) {
  Op::Settings settings(graph, "", {});
  settings.executionContext = decideExecutionContext(graph);

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

void addClipByNorm(Tensor *grad, Tensor *clipFactor, Graph &graph) {
  auto &ir = graph.getIr();

  Op::Settings settings(graph, "", {});
  settings.executionContext = decideExecutionContext(graph);

  Op *mulOp = graph.createOp<MulOp>(Onnx::AiOnnx::OpSet6::Mul, settings);
  transferBaseProperties(grad->consumers.getOps().at(0), mulOp);

  auto clipFactorId = clipFactor->id;

  if (mulOp->hasVirtualGraphId() &&
      mulOp->getVirtualGraphId() != clipFactor->getVirtualGraphId()) {
    auto destIpu          = mulOp->getVirtualGraphId();
    auto copiedClipFactor = createCopyOnVGraph(clipFactor, destIpu, graph);
    clipFactorId          = copiedClipFactor->id;
  }

  auto clippedGradId =
      logging::format("{}_{}", getBaseTensorId(grad->id), "clipped");

  mulOp->connectInTensor(MulOp::getArg0InIndex(), grad->id);
  mulOp->connectInTensor(MulOp::getArg1InIndex(), clipFactorId);
  mulOp->createAndConnectOutTensor(
      MulOp::getOutIndex(), ir.createIntermediateTensorId(clippedGradId));
  mulOp->setup();

  std::stringstream ss;
  ss << "Consumers of " << grad->id << " are:";
  for (auto op : grad->consumers.getOps()) {
    ss << "\n  " << op->str();
  }
  logging::debug("{}", ss.str());

  for (auto op : grad->consumers.getOps()) {
    if (op != mulOp && !op->isConvertibleTo<ReduceSumSquareOp>()) {
      auto indices = op->input->indices(grad);
      for (auto idx : indices) {
        op->disconnectInTensor(idx);
        op->connectInTensor(idx, mulOp->outId(MulOp::getOutIndex()));
      }
    }
  }
}

void addClipByNorms(std::vector<Tensor *> grads,
                    Tensor *clipFactor,
                    Graph &graph) {
  for (auto grad : grads) {
    addClipByNorm(grad, clipFactor, graph);
  }
}

// clip_factor = clip_norm / max(clip_norm, global_norm)
Tensor *createClipFactor(Tensor *globalNorm, Tensor *clipNorm, Graph &graph) {
  auto &ir = graph.getIr();

  Op::Settings settings(graph, "", {});
  settings.executionContext = decideExecutionContext(graph);

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

std::vector<Tensor *> getGrads(Graph &graph,
                               const std::vector<TensorId> &weightIds) {
  std::vector<Tensor *> result;

  auto varUpdates = getVarUpdates(graph, weightIds);
  for (auto op : varUpdates) {
    if (op->isConvertibleTo<SGD0VarUpdateOp>() ||
        op->isConvertibleTo<SGD1VarUpdateOp>()) {
      auto grad = op->inTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());
      result.push_back(grad);
    } else if (op->isConvertibleTo<AdamVarUpdateOp>()) {
      auto adamUpdater =
          op->inTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex())
              ->getProducer();
      if (!adamUpdater->isConvertibleTo<AdamUpdaterOp>()) {
        throw internal_error("This should be a AdamUpdaterOp.");
      }
      auto accl1 = adamUpdater->inTensor(AdamUpdaterOp::getAccl1InIndex())
                       ->getProducer();
      if (!accl1->isConvertibleTo<AccumulateOp>()) {
        throw internal_error("These should be AccumulateOps.");
      }
      auto scaler =
          accl1->inTensor(AccumulateOp::getUpdaterInIndex())->getProducer();
      if (scaler->isConvertibleTo<ScaledAddOp>()) {
        auto grad = scaler->inTensor(ScaledAddOp::getArg0InIndex());
        result.push_back(grad);
      } else if (scaler->isConvertibleTo<MulOp>()) {
        auto grad = scaler->inTensor(MulOp::getArg0InIndex());
        result.push_back(grad);
      } else if (scaler->isConvertibleTo<ScaleOp>()) {
        auto grad = scaler->inTensor(ScaleOp::getInIndex());
        result.push_back(grad);
      } else {
        throw internal_error("Unexpected op type {}", scaler->str());
      }
    } else {
      throw internal_error("Unable to handle op {}", op->str());
    }
  }

  return result;
}

void clipWeightGradientsByNorm(const std::vector<TensorId> &weightIds,
                               float maxNorm,
                               Graph &graph) {
  auto grads = getGrads(graph, weightIds);

  std::vector<Tensor *> gradNorms;
  for (auto grad : grads) {
    auto gradNorm = addReduceSumSquare(grad, graph);
    gradNorms.push_back(gradNorm);
  }

  auto &ir   = graph.getIr();
  auto &opts = ir.getSessionOptions();
  if (opts.enablePipelining) {
    gradNorms = copyToSameVGraph(gradNorms, 0, graph);
  }

  auto globalNorm = createGlobalNorm(gradNorms, graph);
  auto clipNorm   = createClipNorm(maxNorm, graph);
  auto clipFactor = createClipFactor(globalNorm, clipNorm, graph);
  addClipByNorms(grads, clipFactor, graph);
}

} // namespace

std::size_t ClipWeightGradientsByNorm::id() {
  return typeid(ClipWeightGradientsByNorm).hash_code();
}

bool ClipWeightGradientsByNorm::apply(Graph &graph) const {
  auto &ir        = graph.getIr();
  auto &optimizer = ir.getOptimizer();
  auto &opts      = ir.getSessionOptions();

  if (opts.enablePipelining && opts.accumulateOuterFragmentSettings.schedule ==
                                   AccumulateOuterFragmentSchedule::Serial) {
    throw error(
        "Incompatible accumulateOuterFragmentSchedule used with gradient "
        "clipping, SessionOptions::accumulateOuterFragmentSettings.schedule "
        "can not be set to AccumulateOuterFragmentSchedule::Serial");
  }

  for (auto &clipGroup : optimizer.getClipNormSettings()) {
    clipWeightGradientsByNorm(clipGroup.weightIds, clipGroup.maxNorm, graph);
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new ClipWeightGradientsByNorm);
} // namespace

} // namespace popart
