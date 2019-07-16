#include <limits>
#include <poponnx/error.hpp>
#include <poponnx/graph.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/logging.hpp>
#include <poponnx/op/concat.hpp>
#include <poponnx/op/flatten.hpp>
#include <poponnx/op/gradientaccl.hpp>
#include <poponnx/op/varupdate.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/tensors.hpp>
#include <poponnx/topocons.hpp>
#include <poponnx/vertex.hpp>

#include <poponnx/transforms/gradient_accumulation.hpp>

namespace poponnx {

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
    if (op->isConvertibleTo<VarUpdateOp>()) {
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

      // Put it in the backward pass (from loss)
      acclOp->fromLoss = PathFromLoss::Yes;
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

} // namespace poponnx
