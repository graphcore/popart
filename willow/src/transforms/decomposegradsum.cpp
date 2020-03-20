// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/add.hpp>
#include <popart/op/init.hpp>
#include <popart/op/sum.hpp>
#include <popart/opmanager.hpp>
#include <popart/transforms/decomposegradsum.hpp>

namespace popart {

GradPartial::GradPartial(Tensor *t_, std::vector<Op *> pathFromLoss_)
    : t(t_), pathFromLoss(pathFromLoss_) {}

bool GradPartial::operator<(const GradPartial &other) const {
  Op *op      = pathFromLoss.back();
  Op *otherOp = other.pathFromLoss.back();

  // Compare factors to determine the optimal order for the addition tree.
  // Consider annotations that imply an order
  // (pipeline stage, pingpong phase, batch serialized phase)
  // If an attribute is not set, assume that the Op comes before any of the
  // Ops that have the attribute set by using -2
  // (possible because -2 is not used as an actual phase index).

  // TODO(T17524): Abstract inferring operation order so that this
  // transform does not require knowledge of the attributes
  std::tuple<PipelineStage, PingPongPhase, BatchSerializedPhase, size_t> order(
      op->hasPipelineStage() ? op->getPipelineStage() : -2,
      op->hasPingPongPhase() ? op->getPingPongPhase() : -2,
      op->hasBatchSerializedPhase() ? op->getBatchSerializedPhase() : -2,
      pathLengthFromLoss());

  std::tuple<PipelineStage, PingPongPhase, BatchSerializedPhase, size_t>
      otherOrder(otherOp->hasPipelineStage() ? otherOp->getPipelineStage() : -2,
                 otherOp->hasPingPongPhase() ? otherOp->getPingPongPhase() : -2,
                 otherOp->hasBatchSerializedPhase()
                     ? otherOp->getBatchSerializedPhase()
                     : -2,
                 other.pathLengthFromLoss());

  return order < otherOrder;
}

// Consider the paths from loss grad (L') to the three grad partials of a
// gradSum op, {gp0, gp1, gp2}:
// L'->A->B*->C*->gp0                *-count = 2
// L'->A->B*->D->E*->F->G->H*->gp1   *-count = 3
// L'->A->B*->D->E*->I*->J*->gp2     *-count = 4
// where ops with * have a path from toLoss == PathToLoss::Yes

// Design decision: assume that the fewer the number the * ops on the path
// of the grad partial, the earlier it is possible to schedule the addition
// of the grad partial. This relies on the assumption that there is only a
// single path from tensor, T, to the backwards pass, for each consumer of T
// in the forwards pass with a path to the loss.
size_t GradPartial::pathLengthFromLoss() const {
  size_t length = 0;
  for (Op *op : pathFromLoss) {
    for (Tensor *input : op->input->tensors()) {
      if (!input->hasProducer()) {
        continue;
      } else if (input->getProducer()->toLoss == PathToLoss::Yes) {
        length++;
        break;
      }
    }
  }
  return length;
}

std::size_t DecomposeGradSum::id() {
  return typeid(DecomposeGradSum).hash_code();
}

std::vector<Op *>
DecomposeGradSum::getDecomposableGradSumOps(const Graph &graph) const {
  std::vector<Op *> decomposableGradSumOps;
  // An op in the graph is deemed a decomposable GradSumOp if:
  // 1. it is a SumOp
  // 2. its name contains Ir::getGradSumOpNamePrefix()
  // 3. it produces a tensor with an id that contains reservedGradientPrefix()
  // 4. it has a path from the loss
  // 5. it consumes >2 ActGrad tensors
  for (auto &id_op : graph.getOps()) {
    Op *op = id_op.second.get();
    // 1.
    if (op->isConvertibleTo<SumOp>()) {
      // 2.
      if (op->settings.name.find(graph.getIr().getGradSumOpNamePrefix()) !=
          std::string::npos) {
        // 3.
        if (op->outId(SumOp::getOutIndex()).find(reservedGradientPrefix()) !=
            std::string::npos) {
          // 4.
          if (op->outTensor(SumOp::getOutIndex())->fromLoss ==
              PathFromLoss::Yes) {
            auto inputs               = op->input->tensors();
            bool allInputsAreActGrads = true;
            for (Tensor *t : inputs) {
              if (t->tensorType() != TensorType::ActGrad) {
                allInputsAreActGrads = false;
              }
            }
            // 5.
            if (inputs.size() > 2 && allInputsAreActGrads) {
              decomposableGradSumOps.push_back(op);
            }
          }
        }
      }
    }
  }

  return decomposableGradSumOps;
}

// A cycles-for-liveness optimization. Turn gradient summations into
// a schedulable tree of additions such that partial gradients do not
// have to wait until the point of summation of all gradient partials
// to become no longer live.
//
// Consider the model:
// in0 -
//       \
//        Matmul0 - Matmul1 - Matmul2 - loss
//       /          /          /
//  w0 ------------------------
//
// In the backwards pass:
//
// loss_grad
//    |
//    |- Matmul2Grad_rhs ------------- gp0 -    w0 - -
//    |_ Matmul2Grad_lhs                    \         |
//          |                                \        .
//          |- Matmul1Grad_rhs ------- gp1 - Sum -- VarUpdate
//          |_ Matmul1Grad_lhs               /
//                |                         /
//                 - Matmul0Grad_lhs - gp2 -
//
// Observe that the gradient partials, gp0-1, must stay live until gp2
// becomes available so the sum can be computed.
//
// Now observe how the below transform can reduce sum liveness of the
// partial gradient tensors:
//
// loss_grad            InitOp(zero) - gpi -
//    |                                     \
//    |- Matmul2Grad_rhs ------------- gp0 - Add
//    |_ Matmul2Grad_lhs                      \
//          |                                  \
//          |- Matmul1Grad_rhs ------- gp1 ---- Add    w0 - -
//          |_ Matmul1Grad_lhs                   \          |
//                |                               \         .
//                 - Matmul0Grad_lhs - gp2 ------- Add  -- VarUpdate

bool DecomposeGradSum::apply(Graph &graph) const {
  for (Op *gradSumOp : getDecomposableGradSumOps(graph)) {
    logging::debug("Decomposing gradient sum op '{}' into a tree off additions "
                   "of its inputs",
                   gradSumOp->str());
    std::vector<Tensor *> gradPartialTensors = gradSumOp->input->tensors();
    std::vector<GradPartial> gradPartials;

    for (Tensor *gradPartialTensor : gradPartialTensors) {
      std::vector<Op *> pathToLoss;
      Op *pathEnd = gradPartialTensor->getProducer();

      // 1) For each partial grad input to the gradSumOp, get the path
      //    back to thier common loss gradient
      while (pathEnd && pathEnd->fromLoss == PathFromLoss::Yes) {
        // Add the pathEnd to the path
        pathToLoss.push_back(pathEnd);

        // Get the next Op backwards towards the loss.
        // Let's only explore the first valid extension
        // to our path that we find. This is valid, as we
        // are doing the same in all cases, and are only
        // concerned with commonality of paths
        auto inputTensors = pathEnd->input->tensors();
        pathEnd           = nullptr;
        for (Tensor *t : inputTensors) {
          if (!t->hasProducer()) {
            break;
          }
          pathEnd = t->getProducer();
          if (pathEnd->fromLoss == PathFromLoss::Yes) {
            break;
          }
        }
      }
      std::reverse(pathToLoss.begin(), pathToLoss.end()); // Now 'pathFromLoss'
      gradPartials.push_back(GradPartial(gradPartialTensor, pathToLoss));
    }

    // 2) Choose the order in which to sum the grad partials optimially,
    //    based on their path lengths from the loss.
    std::vector<GradPartial> partialsSumOrder = gradPartials;
    std::sort(partialsSumOrder.begin(), partialsSumOrder.end());

    logging::debug("  Partials sum order:");
    for (GradPartial gradPartial : partialsSumOrder) {
      logging::debug("    {}", gradPartial.t->id);
    }

    // Remove the old Grad Sum op
    Tensor *gradSum = gradSumOp->output->tensor(0);
    gradSumOp->disconnectAllInputs();
    gradSumOp->disconnectAllOutputs();
    graph.eraseOp(gradSumOp->id);

    // Now replace with a series of Adds
    std::vector<Op *> addOps;

    // Create InitOp to produce the initial gradient partial input to the
    // addition tree, gpi. It inherits settings from the producer of the first
    // gradient partial
    Op::Settings initSettings =
        partialsSumOrder.front().t->getProducer()->settings;
    initSettings.name = gradSum->id + "_InitOp";
    auto init         = std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                         partialsSumOrder.front().t->info,
                                         TensorType::ActGrad,
                                         InitType::ZERO,
                                         initSettings);
    OpId initOpId     = graph.moveIntoGraph(std::move(init));
    Op *initOp        = graph.getOps()[initOpId].get();
    TensorId gradSumInit = gradSum->id + "_init";
    initOp->createAndConnectOutTensor(InitOp::getOutIndex(), gradSumInit);

    // Since initOp needs to be scheduled post-loss,
    // but has no path from loss, we need to force
    // PathToLoss::No, PathFromLoss::Yes
    initOp->toLoss   = PathToLoss::No;
    initOp->fromLoss = PathFromLoss::Yes;
    initOp->setup();
    TensorId addLhsId = gradSumInit;

    // Is this decomposition part of a batch serialisation?
    bool batchSerialized = false;

    for (size_t i = 0; i < gradPartials.size(); i++) {
      GradPartial gradPartial             = partialsSumOrder.at(i);
      std::unique_ptr<popart::Op> gradAdd = OpManager::createOp(
          Domain::ai_onnx,
          "Add",
          graph.getIr().getOpSetVersionFromModel(Domain::ai_onnx),
          graph,
          "GradAdd" + std::to_string(i));

      OpId opId = graph.moveIntoGraph(std::move(gradAdd));
      Op *op    = graph.getOps()[opId].get();
      op->connectInTensor(AddOp::getArg0InIndex(), addLhsId);
      op->connectInTensor(AddOp::getArg1InIndex(), gradPartial.t->id);
      if (i == gradPartials.size() - 1) {
        // The final summed gradient tensor - it already exists in the Ir
        op->connectOutTensor(AddOp::getOutIndex(), gradSum->id);
      } else {
        TensorId partSummedId = gradSum->id + "_" + std::to_string(i);
        op->createAndConnectOutTensor(AddOp::getOutIndex(), partSummedId);
        // For the next Add
        addLhsId = partSummedId;
      }

      // Gradient accumulator needs the same tensor layout as the gradient.
      // Allow the AddOp to propagate the tensor layout from the gradient
      // to the InitOp output:
      op->settings.inferTensorMappingToFrom.insert(
          {AddOp::getArg0InIndex(), AddOp::getArg1InIndex()});

      op->optionallySetVGraphIdFromMaxOfInputProducers();
      op->optionallySetPipelineStageFromMaxOfInputProducers();
      op->optionallySetPingPongPhaseFromMaxOfInputProducers();
      op->optionallySetBatchSerializedPhaseFromMaxOfInputProducers();
      op->setup();
      op->toLoss   = PathToLoss::No;
      op->fromLoss = PathFromLoss::Yes;
      batchSerialized |= op->hasBatchSerializedPhase();
    }

    if (batchSerialized) {
      initOp->setBatchSerializedPhase(-1);
    }
  }

  return true;
}

namespace {
// DecomposeGradSum
bool init = Transform::registerTransform(new DecomposeGradSum());
} // namespace

} // namespace popart
