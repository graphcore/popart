// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/gradgrowerop.hpp>

#include <popart/bwdgraphinfo.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/pbwrap.hpp>
#include <popart/tensor.hpp>

namespace popart {

GradGrowerOp::GradGrowerOp(AutodiffIrInterface &dep)
    : GradGrowerOpInterface(), AutodiffHelper(dep) {}

std::vector<Op *>
GradGrowerOp::growGradOps(Op *nonGradOp,
                          const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) {

  logging::ir::debug("Growing grad ops for {}", nonGradOp->str());

  PipelineStage maxPipelineStage = 0;
  if (dep.get().getSessionOptions().enablePipelining) {
    // the last fwd pass pipeline stage is also the first bwd pass pipeline
    // stage.
    maxPipelineStage = dep.get().getFinalLossPipelineStage() * 2;
  }

  OpId nonGradOpId  = nonGradOp->id;
  auto calledGraphs = nonGradOp->getCalledGraphs();
  if (!nonGradOp->getCalledGraphs().empty()) {
    nonGradOp->setCalledSubgraphGradInfo(calledGraphsGradInfo);
  }
  auto backOps = nonGradOp->getGradOps();
  if (backOps.size() < 1) {
    logging::ir::debug("Cannot get gradients for {}", nonGradOp->str());
  }
  std::vector<Op *> gradOps;
  for (auto &upop : backOps) {
    Op *gradOp    = upop.get();
    OpId gradOpId = dep.get().getMainGraph().moveIntoGraph(std::move(upop));

    // Reset priority, since fwd priority should not influence bwd priority
    //
    // TODO: Uncomment this. This prevented explicit priorities on certain
    // gradient ops being set which was necessary as a short term fix for
    // sharded training regressions seen in T17036. This could be replaced
    // once explicit priorities are no longer needed for this purpose. T17311
    // should fix this.
    //
    // gradOp->settings.schedulePriority = 0.0;

    if (gradOp->hasExecutionPhase()) {
      // Remap from forward to backward execution phase
      gradOp->setExecutionPhase(
          2 * dep.get().getSessionOptions().executionPhaseSettings.phases - 2 -
          gradOp->getExecutionPhase());
    }

    if (nonGradOp->settings.recomputeType == RecomputeType::Recompute &&
        dep.get().getSessionOptions().autoRecomputationEnabled() &&
        dep.get().getSessionOptions().executionPhaseSettings.phases < 2) {
      throw error("Grad Ops should be grown before recompute annotation");
    }

    // No gradOp should be of type Recompute.
    gradOp->settings.recomputeType = RecomputeType::Checkpoint;

    if (nonGradOp->hasPipelineStage()) {
      gradOp->setPipelineStage(maxPipelineStage -
                               nonGradOp->getPipelineStage());
    }

    // connect inputs of gradOp
    {
      // inputs to gradOp (to populate in this scope):
      std::map<int, std::string> m_inputs;
      auto isInputOptional = [](Op *op, InIndex i) {
        auto optionalInputs = op->optionalInputs();
        return optionalInputs.find(i) != optionalInputs.end();
      };
      for (auto &inOutMapper : gradOp->gradInputInfo()) {

        int indexGrad     = inOutMapper.iGrad;
        int indexFwd      = inOutMapper.iNonGrad;
        GradOpInType type = inOutMapper.type;

        // the input at index 'indexGrad' to gradOp is
        switch (type) {
        //  (1) the INPUT at index 'indexFwd' of nonGradOp
        case GradOpInType::In: {
          if (nonGradOp->input->hasIndex(indexFwd)) {
            m_inputs[indexGrad] = nonGradOp->input->tensor(indexFwd)->id;
          } else if (isInputOptional(nonGradOp, indexFwd)) {
            m_inputs[indexGrad] = TensorId();
          } else {
            throw error(
                "Invalid configuration of gradOp {}. nonGradOp ({}) INPUT {} "
                "is not marked as optional, but is not defined",
                gradOp->debugName(),
                nonGradOp->debugName(),
                indexFwd);
          }
          break;
        }

        //  (2) the OUTPUT at index 'indexFwd' of nonGradOp
        case GradOpInType::Out: {
          if (!nonGradOp->output->hasIndex(indexFwd)) {
            throw error("Invalid configuration of gradOp {}. nonGradOp ({}) "
                        "OUTPUT {} is not defined ",
                        gradOp->debugName(),
                        nonGradOp->debugName(),
                        indexFwd);
          }
          m_inputs[indexGrad] = nonGradOp->output->tensor(indexFwd)->id;
          break;
        }

        //  (3) the GRADIENT of the OUTPUT
        //      at index 'indexFwd' of nonGradOp.
        case GradOpInType::GradOut: {
          if (!nonGradOp->output->hasIndex(indexFwd)) {
            throw error("Invalid configuration of gradOp {}. nonGradOp ({}) "
                        "OUTPUT {} is not defined ",
                        gradOp->debugName(),
                        nonGradOp->debugName(),
                        indexFwd);
          }

          auto gradTensorId =
              getGradId(nonGradOp->output->tensor(indexFwd)->id);
          if (dep.get().getMainGraph().getTensors().contains(
                  gradTensorId, gradOp->getScope())) {
            m_inputs[indexGrad] = gradTensorId;
          } else {
            m_inputs[indexGrad] = TensorId();
          }
          break;
        }
        }
      }

      dep.get().getMainGraph().connectInputs(InputMapWrapper(m_inputs),
                                             gradOpId);
    }

    // connect outputs of gradOp
    {
      std::vector<TensorId> v_outputs;
      for (auto out_in : gradOp->gradOutToNonGradIn()) {
        int gradOut   = out_in.first;
        int nonGradIn = out_in.second;

        if (v_outputs.size() < gradOut + 1) {
          v_outputs.resize(gradOut + 1, TensorId());
        }

        if (nonGradOp->input->hasIndex(nonGradIn)) {
          TensorId inId      = nonGradOp->input->tensor(nonGradIn)->id;
          TensorId outId     = getEdgeGradId(nonGradOpId, nonGradIn);
          v_outputs[gradOut] = outId;
        }
      }
      dep.get().getMainGraph().connectOutputs(OutputVecWrapper(v_outputs),
                                              gradOpId);
    }
    gradOp->setup();

    // note, as the outputs of gradOp are edge-grad-tensors and not
    // edge-grads, we do not need to match them to non-grad tensors.
    gradOps.push_back(gradOp);
  }

  return gradOps;
}

} // namespace popart
