// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/patterns/lambserialisedweight.hpp>

#include <patterns/tiedgatherutils/tgutils.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/adamvarupdate.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/op/lamb.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/sum.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/patterns/patterns.hpp>

namespace popart {

namespace {

bool isProducedBySlice(const Tensor *t) {
  return t->hasProducer() && t->getProducer()->isConvertibleTo<BaseSliceOp>();
}

bool isConsumedByAdd(const Tensor *t) {
  for (auto cons : t->consumers.getOps()) {
    if (cons->isConvertibleTo<SumOp>()) {
      return true;
    }
  }
  return false;
}

} // namespace

bool LambSerialisedWeightPattern::matches(Op *op) const {
  auto &ir = op->getIr();
  // Don't run in inference
  if (!ir.canTrain()) {
    return false;
  }
  if (!ir.hasDecomposedOptimizers()) {
    return false;
  }

  auto lambsq = dynamic_cast<LambSquareOp *>(op);
  if (lambsq) {
    auto wSlice = tgutil::maybeTraverseProducer<ReplicatedReduceScatterOp>(
        ReplicatedReduceScatterOp::getInIndex(),
        op->inTensor(LambSquareOp::getInIndex()));
    bool onlyASlice = isProducedBySlice(wSlice);
    // Check we haven't already applied the pattern.
    return onlyASlice &&
           !isConsumedByAdd(lambsq->outTensor(LambSquareOp::getOutIndex()));
  }
  return false;
}

std::vector<const Tensor *> LambSerialisedWeightPattern::touches(Op *) const {
  return {};
}

bool LambSerialisedWeightPattern::apply(Op *op) const {
  logging::pattern::trace("[LambSerialisedWeight] Applying to Op: {}",
                          op->debugName());
  auto &graph = op->getGraph();

  // To fix R1:
  //  1. Find root weight
  //  2. Find all LambSquareOp consumers
  //  3. Insert a SumOp between the consumers of LambSquareOp->outTensor(0) and
  //  their consumers

  // (1)
  auto rootWeight = tgutil::getVariable(
      tgutil::maybeTraverseProducer<ReplicatedReduceScatterOp>(
          ReplicatedReduceScatterOp::getInIndex(),
          op->inTensor(LambSquareOp::getInIndex())));
  // (2)
  auto r1LambOps =
      tgutil::findAllConsumers<LambSquareOp,
                               ExecutionContext::AccumulateOuterFragment>(
          rootWeight);

  // Only Sum if there is more than one Op
  if (r1LambOps.size() <= 1) {
    return false;
  }

  std::vector<TensorId> r1Inputs(r1LambOps.size());
  std::transform(
      r1LambOps.begin(), r1LambOps.end(), r1Inputs.begin(), [](auto lambsq) {
        return lambsq->outId(LambSquareOp::getOutIndex());
      });

  TensorId r1Output = reservedLambR1SqPrefix() + rootWeight->id;

  // (3)
  auto r1SumOp = insertSumOp(graph, r1Inputs, r1Output, op, "R1SerialisedSum");

  std::vector<Op *> varUpdatesToSearchForR2;

  // For each of r1Inputs, for each consumer that isn't the sum op,
  // reconnect the consumer to the output of the sum.
  // If the consumer is an AdamVarUpdate, add to varUpdatesToSearchForR2.

  for (auto lambOp : r1LambOps) {
    auto tensorToReplace = lambOp->outTensor(LambSquareOp::getOutIndex());
    for (auto cons : tensorToReplace->consumers.getOps()) {
      if (cons->id != r1SumOp->id) {
        auto update = tgutil::searchConsumersFor<
            AdamVarUpdateOp,
            ExecutionContext::AccumulateOuterFragment>(tensorToReplace);

        for (auto in_index : cons->input->indices(tensorToReplace)) {
          cons->disconnectInTensor(tensorToReplace);
          cons->connectInTensor(in_index, r1Output);
        }
        if (update != nullptr) {
          varUpdatesToSearchForR2.push_back(update);
        }
      }
    }
  }

  // To fix R2:
  //  1. Start from the AdamVarUpdateOps from fixing R1
  //  2. Find all LambSquareOps in their producers on
  //  AdamVarUpdate::getR2InIndex()
  //  3. Insert a SumOp between the consumers of LambSquareOp->outTensor(0) and
  //  their consumers

  std::vector<Op *> r2LambOps;
  // (1)
  for (auto varUpdateOp : varUpdatesToSearchForR2) {
    // (2)
    auto r2_tensor =
        varUpdateOp->inTensor(AdamVarUpdateOp::getLambR2SqInIndex());
    auto r2_op =
        tgutil::searchProducersFor<LambSquareOp,
                                   ExecutionContext::AccumulateOuterFragment>(
            r2_tensor);
    if (r2_op == nullptr) {
      // If this pattern matched, we have to be able to apply it for
      // correcntess, so we have to error if we cannot proceed.
      throw error("[LambSerialisedWeight] Could not find the R2 LambSquareOp "
                  "for AdamVarUpdate {}",
                  varUpdateOp->debugName());
    }
    r2LambOps.push_back(r2_op);
  }

  // (3)
  std::vector<TensorId> r2Inputs(r2LambOps.size());
  std::transform(
      r2LambOps.begin(), r2LambOps.end(), r2Inputs.begin(), [](auto lambsq) {
        return lambsq->outId(LambSquareOp::getOutIndex());
      });

  TensorId r2Output = reservedLambR2SqPrefix() + rootWeight->id;

  // (3)
  auto r2SumOp = insertSumOp(graph, r2Inputs, r2Output, op, "R2SerialisedSum");

  for (auto lambOp : r2LambOps) {
    auto tensorToReplace = lambOp->outTensor(LambSquareOp::getOutIndex());
    for (auto cons : tensorToReplace->consumers.getOps()) {
      if (cons->id != r2SumOp->id) {
        for (auto in_index : cons->input->indices(tensorToReplace)) {
          cons->disconnectInTensor(tensorToReplace);
          cons->connectInTensor(in_index, r2Output);
        }
      }
    }
  }

  return true;
}

SumOp *LambSerialisedWeightPattern::insertSumOp(Graph &graph,
                                                std::vector<TensorId> &inIds,
                                                TensorId outId,
                                                Op *refOp,
                                                std::string debugName) const {
  auto sum = graph.createOp<SumOp>(Onnx::Operators::Sum_8,
                                   Op::Settings(graph, debugName));
  transferBaseProperties(refOp, sum);

  for (unsigned i = 0; i < inIds.size(); i++) {
    sum->connectInTensor(i, inIds[i]);
  }

  sum->createAndConnectOutTensor(SumOp::getOutIndex(), outId);
  sum->setup();

  sum->settings.executionContext = ExecutionContext::AccumulateOuterFragment;
  // SumToAdd pattern uses makeReplacementOpInIr, which does not transfer the
  // ExecutionContext, so we must avoid the pattern.
  sum->settings.excludePatterns.insert("SumToAdd");

  return sum;
}

namespace {
PatternCreator<LambSerialisedWeightPattern>
    tiedGatherer(PreAliasPatternType::LambSerialisedWeight,
                 "LambSerialisedWeight",
                 true,   // On by default
                 false); // Not mandatory
} // namespace

} // namespace popart
