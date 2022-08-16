// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <transforms/autodiff/gradgrowerop.hpp>
#include <utility>
#include <popart/bwdgraphinfo.hpp>
#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/pbwrap.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensors.hpp"
#include "popart/util.hpp"
#include "transforms/autodiff/autodiffhelper.hpp"
#include "transforms/autodiff/autodiffirinterface.hpp"

namespace popart {

GradGrowerOp::GradGrowerOp(AutodiffIrInterface &dep_)
    : GradGrowerOpInterface(), AutodiffHelper(dep_) {}

std::vector<Op *>
GradGrowerOp::growGradOps(Graph &bwdGraph,
                          Op *nonGradOp,
                          const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) {

  logging::ir::debug("Growing grad ops for {}", nonGradOp->str());

  auto &ir = dep.get();

  PipelineStage maxPipelineStage = 0;
  if (ir.getSessionOptions().enablePipelining) {
    maxPipelineStage = ir.getMaxPipelineStage();
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
    OpId gradOpId = bwdGraph.moveIntoGraph(std::move(upop));

    // Reset priority, since fwd priority should not influence bwd priority.
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
          2 * ir.getSessionOptions().executionPhaseSettings.phases - 2 -
          gradOp->getExecutionPhase());
    }

    if (nonGradOp->settings.recomputeType == RecomputeType::Recompute &&
        ir.getSessionOptions().autoRecomputationEnabled() &&
        ir.getSessionOptions().executionPhaseSettings.phases < 2) {
      throw error("Grad Ops should be grown before recompute annotation");
    }

    // No gradOp should be of type Recompute.
    gradOp->settings.recomputeType = RecomputeType::Checkpoint;

    if (nonGradOp->hasPipelineStage()) {
      gradOp->setPipelineStage(maxPipelineStage -
                               nonGradOp->getPipelineStage());
    }

    // To retrieve the inputs we need to connect to the grad op:
    //
    // For inputs that are forward tensors, they will have been cloned into the
    // backward graph already, regardless of whether we did recompute or
    // fwdoutput stitching. Therefore, we use `fwdIdToClonedBwdId` to get the id
    // of the cloned tensor in the bwd graph from the fwd graph. Note, if this
    // is the main graph, then bwd graph and fwd graph will be the same, so this
    // is a nop.
    //
    // For inputs that are upstream gradient tensors, we expect them to have
    // already been created, and retrieve them by using `fwdIdToBwdGradId` on
    // the forward tensor in the fwd graph.

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
            const auto &inIdInFwdGraph = nonGradOp->input->tensor(indexFwd)->id;
            const auto inIdInBwdGraph  = fwdIdToClonedBwdId(
                nonGradOp->getGraph(), bwdGraph, inIdInFwdGraph);
            m_inputs[indexGrad] = inIdInBwdGraph;
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
          const auto &outIdInFwdGraph = nonGradOp->output->tensor(indexFwd)->id;
          const auto outIdInBwdGraph  = fwdIdToClonedBwdId(
              nonGradOp->getGraph(), bwdGraph, outIdInFwdGraph);
          m_inputs[indexGrad] = outIdInBwdGraph;
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

          const auto &nonGradId = nonGradOp->output->tensor(indexFwd)->id;
          const auto bwdGraphGradId =
              fwdIdToBwdGradId(nonGradOp->getGraph(), bwdGraph, nonGradId);

          // Surely this is an error if false (unless optional input)? See
          // D49736 for when it was changed to not be an error.
          // Do we need to pass scope to contains?
          if (bwdGraph.getTensors().contains(bwdGraphGradId,
                                             gradOp->getScope())) {
            m_inputs[indexGrad] = bwdGraphGradId;
          } else {
            m_inputs[indexGrad] = TensorId();
          }
          break;
        }
        }
      }

      bwdGraph.connectInputs(InputMapWrapper(m_inputs), gradOpId);
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

        // To construct the ids of the output tensors of the grad op, we use
        // `getEdgeGradId` then add the scope of the bwd graph.
        if (nonGradOp->input->hasIndex(nonGradIn)) {
          TensorId inId = nonGradOp->input->tensor(nonGradIn)->id;
          TensorId outId =
              addScope(bwdGraph, getEdgeGradId(nonGradOpId, nonGradIn));
          v_outputs[gradOut] = outId;
        }
      }
      bwdGraph.connectOutputs(OutputVecWrapper(v_outputs), gradOpId);
    }
    gradOp->setup();

    // note, as the outputs of gradOp are edge-grad-tensors and not
    // edge-grads, we do not need to match them to non-grad tensors.
    gradOps.push_back(gradOp);
  }

  return gradOps;
}

} // namespace popart
