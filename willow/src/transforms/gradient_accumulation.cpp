#include <limits>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/flatten.hpp>
#include <popart/op/gradientaccl.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/vertex.hpp>

#include <popart/transforms/gradient_accumulation.hpp>

namespace popart {

std::size_t GradientAccumulation::id() {
  return typeid(GradientAccumulation).hash_code();
}

bool GradientAccumulation::apply(Graph &graph) const {
  auto &ir = graph.getIr();

  if (!ir.getSessionOptions().enableGradientAccumulation)
    return true;

  if (!ir.canTrain() && ir.getSessionOptions().enableGradientAccumulation) {
    throw error("Gradient Accumulation only available when training.");
  }

  for (auto &id_op : graph.getOps()) {
    auto &op = id_op.second;
    // Only apply to VarUpdate ops that are not CopyVarUpdate ops that appear
    // in batch norms.
    if (op->isConvertibleTo<VarUpdateOp>() &&
        !(op->isConvertibleTo<CopyVarUpdateOp>())) {
      // Get the gradient tensor id
      auto gradTensorId = op->input->id(VarUpdateOp::getUpdaterInIndex());

      // Create a new Tensor to store accumulations.
      // Then swap the accl and gradient tensors on the varUpdate
      auto acclTensorId      = reservedAccumulationPrefix() + gradTensorId;
      auto acclOutTensorId   = reservedAccumulationOutPrefix() + gradTensorId;
      auto acclResetTensorId = reservedAccumulationResetPrefix() + gradTensorId;

      op->replaceInTensorWithZeros(VarUpdateOp::getUpdaterInIndex(),
                                   acclTensorId);

      logging::transform::trace("Created accumulation Tensor: {}",
                                acclTensorId);

      // Create accl op.
      std::string opName;
      if (!op->getName().empty()) {
        opName = reservedAccumulationPrefix() + op->getName();
      } else {
        opName = "";
      }

      auto acclOp_up = std::make_unique<GradientAcclOp>(
          Onnx::CustomOperators::GradientAccumulation,
          Op::Settings(graph, opName));

      auto acclOp = acclOp_up.get();
      graph.moveIntoGraph(std::move(acclOp_up));
      logging::transform::trace("Created accumulation operation: {}",
                                acclOp->debugName());

      acclOp->connectInTensor(GradientAcclOp::getAcclInIndex(), acclTensorId);
      acclOp->connectInTensor(GradientAcclOp::getGradInIndex(), gradTensorId);
      acclOp->createAndConnectOutTensor(GradientAcclOp::getAcclOutIndex(),
                                        acclOutTensorId);
      if (op->hasVirtualGraphId()) {
        // Inherit virtual graph from parent op
        acclOp->setVirtualGraphId(op->getVirtualGraphId());
      }

      // Add it to train target ops to prevent pruning.
      if (!ir.addToTrainTargetOps(acclOp)) {
        throw error("Could not add {} to train target ops", acclOp->id);
      }

      // Move constraints from varUpdate to accumulator
      graph.topoCons->transfer(op.get(), acclOp);
      // Add constraint to make accumulate before varUpdate
      graph.topoCons->insert(acclOp, op.get());

      acclOp->setup();

      // Create accl reset op.
      std::string resetOpName;
      if (!op->getName().empty()) {
        resetOpName = reservedAccumulationResetPrefix() + op->getName();
      } else {
        resetOpName = "";
      }
      auto resetOp_up = std::make_unique<ResetAcclOp>(
          Onnx::CustomOperators::ResetAccumulation,
          Op::Settings(graph, resetOpName));
      auto resetOp = resetOp_up.get();
      graph.moveIntoGraph(std::move(resetOp_up));
      logging::transform::trace("Created accumulation reset operation: {}",
                                resetOp->debugName());

      if (!ir.addToTrainTargetOps(resetOp)) {
        throw error("Could not add {} to train target ops", resetOp->id);
      }

      resetOp->connectInTensor(ResetAcclOp::getAcclInIndex(), acclTensorId);
      resetOp->createAndConnectOutTensor(ResetAcclOp::getAcclOutIndex(),
                                         acclResetTensorId);
      if (op->hasVirtualGraphId()) {
        // Inherit virtual graph from parent op
        resetOp->setVirtualGraphId(op->getVirtualGraphId());
      }
      op.get()->connectInTensor(VarUpdateOp::getUpdaterInIndex(),
                                acclOutTensorId);

      resetOp->setup();
    }
  }
  return true;
}

namespace {
bool init = Transform::registerTransform(new GradientAccumulation);
}

} // namespace popart
