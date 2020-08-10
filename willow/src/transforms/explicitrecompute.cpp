#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/boundary.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/slice.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/explicitrecompute.hpp>

namespace popart {

using TensorContext = std::tuple<VGraphId, ExecutionPhase, PipelineStage>;

std::size_t ExplicitRecompute::id() {
  return typeid(ExplicitRecompute).hash_code();
}

bool ExplicitRecompute::apply(Graph &graph) const {
  logging::transform::debug("[ExplicitRecompute] Started.");

  auto &ir      = graph.getIr();
  auto schedule = graph.getOpSchedule({});

  auto getContext = [&ir](Op *op) -> TensorContext {
    VGraphId vgid = op->hasVirtualGraphId() ? op->getVirtualGraphId() : -1;
    ExecutionPhase executionPhase =
        (ir.getSessionOptions().executionPhaseSettings.phases > 1 &&
         op->hasExecutionPhase())
            ? op->getExecutionPhase()
            : -1;
    PipelineStage pipelineStage =
        (ir.getSessionOptions().enablePipelining && op->hasPipelineStage())
            ? op->getPipelineStage()
            : -1;
    return TensorContext(vgid, executionPhase, pipelineStage);
  };

  std::map<std::pair<TensorId, TensorContext>, TensorId> recomputedTensorMap;

  for (Op *op : schedule) {
    if (op->settings.recomputeType == RecomputeType::Recompute) {
      // Change every recompute op to checkpoint
      op->settings.recomputeType = RecomputeType::Checkpoint;
      auto clone                 = op->clone();
      auto cloneid               = graph.moveIntoGraph(std::move(clone));

      Op *clone_op = graph.getOp(cloneid);

      if (clone_op->hasExecutionPhase()) {
        // Remap from forward to backward execution phase
        ExecutionPhase recomputePhase =
            2 * ir.getSessionOptions().executionPhaseSettings.phases - 2 -
            clone_op->getExecutionPhase();
        logging::trace(
            "[ExplicitRecompute] Remapping {} execiton phase {} -> {}",
            clone_op->debugName(),
            clone_op->getExecutionPhase(),
            recomputePhase);
        clone_op->setExecutionPhase(recomputePhase);
      }

      // Get context after remapping phases
      auto context = getContext(clone_op);

      clone_op->disconnectAllInputs();
      clone_op->disconnectAllOutputs();
      clone_op->settings.recomputeType = RecomputeType::Recomputed;

      for (auto &in : op->input->tensorMap()) {
        auto recomputedTensor =
            recomputedTensorMap.find({in.second->id, context});
        if (recomputedTensor == recomputedTensorMap.end()) {
          // Not recomputed, consume forward produced tensor
          clone_op->connectInTensor(in.first, in.second->id);
        } else {
          // Recomputed, use recomputed tensor
          clone_op->connectInTensor(in.first, recomputedTensor->second);
        }
      }
      for (auto &out : op->output->tensorMap()) {
        TensorId recomputedId = createRecomputedTensorId(out.second->id);
        recomputedTensorMap[{out.second->id, context}] = recomputedId;
        clone_op->createAndConnectOutTensor(out.first, recomputedId);
      }
      clone_op->setup();

      logging::transform::trace("[ExplicitRecompute] Cloned op {} {} -> {}",
                                clone_op->opid,
                                clone_op->input->getIndexShapeMap(),
                                clone_op->output->getIndexShapeMap());
    }
  }

  // Remap consumer inputs to use recomputed tensor
  for (auto recomputedTensor : recomputedTensorMap) {
    Tensor *tensor = graph.getTensors().get(recomputedTensor.first.first);
    for (Op *consumer : tensor->consumers.getOps()) {
      auto context = getContext(consumer);
      if (((consumer->toLoss == PathToLoss::No &&
            consumer->fromLoss == PathFromLoss::Yes) ||
           consumer->settings.recomputeType == RecomputeType::Recomputed) &&
          context == recomputedTensor.first.second) {
        auto indices = consumer->input->indices(tensor);
        for (auto i : indices) {
          consumer->disconnectInTensor(i, tensor);
          consumer->connectInTensor(i, recomputedTensor.second);
        }
      }
    }
  }

  logging::transform::debug("[ExplicitRecompute] Done.");
  return true;
}

namespace {
// ExplicitRecompute
bool init = Transform::registerTransform(new ExplicitRecompute());
} // namespace

} // namespace popart
