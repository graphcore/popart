// Copyright(c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/patterns/sparseaccumulatepattern.hpp>

#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/gather.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/tensor.hpp>
#include <popart/topocons.hpp>

namespace popart {

bool SparseAccumulatePattern::matches(Op *op) const {
  if (op->isConvertibleTo<GatherGradOp>()) {
    Tensor *gradient = op->outTensor(GatherGradOp::gradOutIndex());

    // If there is more than one consumer, we cannot elide the grad tensor.
    if (gradient->consumers.getTotal() > 1) {
      return false;
    }

    for (Op *consumer : gradient->consumers.getOps()) {
      if (consumer->isConvertibleTo<AccumulateOp>() &&
          !consumer->isConvertibleTo<SparseAccumulateOp>()) {
        const auto accOp = dynamic_cast<AccumulateOp *>(consumer);
        return SparseAccumulateOp::supportsAccumulationType(
            accOp->getAccumulationType());
      }
    }
  }
  return false;
}

std::vector<const Tensor *> SparseAccumulatePattern::touches(Op *) const {
  return {};
}

bool SparseAccumulatePattern::apply(Op *op) const {
  auto &graph = op->getGraph();

  GatherGradOp *gatherGrad = dynamic_cast<GatherGradOp *>(op);
  AccumulateOp *denseAccum = nullptr;

  Tensor *gradient = op->outTensor(GatherGradOp::gradOutIndex());
  for (Op *consumer : gradient->consumers.getOps()) {
    if (consumer->isConvertibleTo<AccumulateOp>() &&
        !consumer->isConvertibleTo<SparseAccumulateOp>()) {
      denseAccum = dynamic_cast<AccumulateOp *>(consumer);
      break;
    }
  }

  if (!denseAccum) {
    throw internal_error(
        "[SparseAccumulatePattern] Applying to GatherGrad `{}` but cannot find "
        "an Accumulate in consumers. Pattern should not have matched.",
        op->debugName());
  }

  auto accumType = denseAccum->getAccumulationType();
  if (!SparseAccumulateOp::supportsAccumulationType(accumType)) {
    throw internal_error(
        "[SparseAccumulatePattern] Applying to Accumulate `{}` with "
        "unsupported AccumulationType {}. Pattern should not have matched.",
        denseAccum->debugName(),
        denseAccum->getAccumulationType());
  }

  auto sparseAccum = graph.createOp<SparseAccumulateOp>(
      accumType,
      denseAccum->getFactor(),
      gatherGrad->getAxis(),
      Op::Settings(graph, denseAccum->name() + "_accumulate"));
  transferBaseProperties(denseAccum, sparseAccum);

  // dX -> GatherGrad(indices) -> dW -> Accumulate(w; factor) -> updatedW
  const auto dX      = gatherGrad->inId(GatherGradOp::gradInIndex());
  const auto indices = gatherGrad->inId(GatherGradOp::indicesInIndex());
  const auto dW      = gatherGrad->outId(GatherGradOp::gradOutIndex());
  const auto w       = denseAccum->inId(AccumulateOp::getVarToUpdateInIndex());
  const auto updatedW =
      denseAccum->outId(AccumulateOp::getUpdatedVarOutIndex());

  // Inputs

  // VarToUpdate -> denseAccum VarToUpdate
  sparseAccum->connectInTensor(SparseAccumulateOp::getVarToUpdateInIndex(), w);

  // Updater -> gatherGrad gradIn
  sparseAccum->connectInTensor(SparseAccumulateOp::getUpdaterInIndex(), dX);

  // Factor -> denseAccum Factor
  if (!denseAccum->getFactor().isConst()) {
    const auto factor = denseAccum->inId(AccumulateOp::getFactorInIndex());
    sparseAccum->connectInTensor(SparseAccumulateOp::getFactorInIndex(),
                                 factor);
  }

  // Indices -> gatherGrad Indices
  sparseAccum->connectInTensor(SparseAccumulateOp::getIndicesInIndex(),
                               indices);

  // Transfer TopoCons
  graph.topoCons->transfer(gatherGrad, sparseAccum);
  graph.topoCons->transfer(denseAccum, sparseAccum);

  // Delete the replaced ops
  denseAccum->disconnectAllInputs();
  denseAccum->disconnectAllOutputs();
  graph.eraseOp(denseAccum->id);

  gatherGrad->disconnectAllInputs();
  gatherGrad->disconnectAllOutputs();
  graph.eraseOp(gatherGrad->id);

  // Outputs
  // UpdatedVar -> denseAccum UpdatedVar
  sparseAccum->connectOutTensor(SparseAccumulateOp::getUpdatedVarOutIndex(),
                                updatedW);

  // Remove the gatherGrad output
  graph.getTensors().remove(dW);

  // Finalise sparse op
  sparseAccum->setup();

  return true;
}

namespace {
PatternCreator<SparseAccumulatePattern>
    patternCreator("SparseAccumulate",
                   true, // Enabled by default
                   false // Not mandatory
    );
} // namespace

} // namespace popart
