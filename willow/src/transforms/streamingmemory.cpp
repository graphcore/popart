// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/boundary.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/lamb.hpp>
#include <popart/op/remote.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/streamingmemory.hpp>
#include <popart/transforms/streamingmemoryopinserter.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

std::size_t StreamingMemory::id(int pass) {
  return typeid(StreamingMemory).hash_code() + pass;
}

// The cost of an Op, simplified to only account for weights
float StreamingMemory::costFn(Op *op) const {
  float w_weights = 1.f;
  float total     = 0;

  for (auto map : op->input->indicesMap()) {
    if (map.first->tensorType() == TensorType::Variable ||
        map.first->tensorType() == TensorType::Const) {
      total += w_weights * static_cast<float>(map.first->info.nbytes());
    }
  }

  logging::trace("Op {} cost {}", op->opid, total);

  return total;
}

void StreamingMemory::verifyExecutionPhases(Graph &graph) const {
  // Verify execution phase annotations
  for (auto &op : graph.getOps()) {
    Op *op0 = op.second.get();
    if (op0->settings.executionContext == ExecutionContext::Normal) {
      if (op0->hasExecutionPhase()) {
        auto phase0 = op0->getExecutionPhase();
        for (Tensor *input : op0->input->tensors()) {
          if (input->hasProducer() &&
              input->getProducer()->settings.executionContext ==
                  ExecutionContext::Normal &&
              input->getProducer()->hasExecutionPhase()) {
            Op *op1     = input->getProducer();
            auto phase1 = op1->getExecutionPhase();
            if (phase1 > phase0) {
              throw error("[StreamingMemory] Op {} {} (I/O) before op {} {}, "
                          "but execution phases disagree ({} vs. {})",
                          op1->id,
                          op1->debugName(),
                          op0->id,
                          op0->debugName(),
                          phase1,
                          phase0);
            }
          }
        }
        for (Op *op1 : graph.topoCons->getBefores(op0)) {
          if (op1->settings.executionContext == ExecutionContext::Normal) {
            auto phase1 = op1->getExecutionPhase();
            if (phase1 > phase0) {
              logging::transform::warn(
                  "[StreamingMemory] Op {} {} (topologically) before op {} {}, "
                  "but execution phases disagree ({} vs. {}). "
                  "Removing constraint.",
                  op1->id,
                  op1->debugName(),
                  op0->id,
                  op0->debugName(),
                  phase1,
                  phase0);
              graph.topoCons->remove(op1, op0);
            }
          }
        }
      }
    }
  }
}

bool StreamingMemory::apply(Graph &graph) const {

  auto &ir                    = graph.getIr();
  auto &sessionOptions        = ir.getSessionOptions();
  const int replicationFactor = sessionOptions.enableReplicatedGraphs
                                    ? sessionOptions.replicatedGraphCount
                                    : 1;
  const int total_num_ipus = ir.getDeviceInfo()->getNumIpus();
  const int num_ipus       = total_num_ipus / replicationFactor;

  // const auto training = ir.canTrain();
  const auto num_stages = sessionOptions.executionPhaseSettings.stages;
  const auto num_phases = sessionOptions.executionPhaseSettings.phases;

  // Tensor remote store/load inserted in the third ping-pong pass only
  StreamingMemoryOpInserter opInserter{
      graph, replicationFactor, num_stages, num_phases};

  if (graph.getOps().size() == 0 || num_phases < 2) {
    return false;
  }

  logging::transform::debug(
      "[StreamingMemory] Execution scheme with {} phases, {} ({}) ipus",
      num_phases,
      num_ipus,
      total_num_ipus);

  auto schedule = graph.getOpSchedule({});

  if (pass == 1 || pass == 2) {
    for (Tensor *tensor : graph.getTensors().getOfType(TensorType::Variable)) {
      // The mechanism by which we handle offloaded (off-chip) tensors of type
      // TensorType::Variable is setting a flag in tensorLocationInfo.
      TensorLocation tensorLocation =
          opInserter.determineTensorLocation(tensor);
      tensor->tensorLocationInfo.setRemote(tensorLocation.isRemote());
      logging::transform::debug("[StreamingMemory] Set Variable {} to {}.",
                                tensor->id,
                                tensorLocationToStr(tensorLocation));
    }
  }

  if (pass == 1) {
    float cumulative_cost = 0.f;

    for (Op *op : schedule) {
      cumulative_cost += costFn(op);
    }

    float cost_per_phase = cumulative_cost / static_cast<float>(num_phases);

    // Greedy claiming of ops per phase according to execution schedule
    // TODO T10602: Find better graph-cut algorithm for phase splitting
    ExecutionPhase phase = 0;
    float phase_cost     = 0;
    for (Op *op : schedule) {

      auto cost = costFn(op);
      // Every phase should handle at least some cost, but not too much
      if (phase_cost > 0 && phase_cost + cost > cost_per_phase &&
          phase < (num_phases - 1)) {
        ++phase;
        phase_cost = 0;
      }
      phase_cost += cost;

      bool has_phase = op->hasExecutionPhase();

      if (has_phase && op->getExecutionPhase() >= num_phases) {
        throw error(
            "[StreamingMemory] Maximum phase is {}, but op {} has phase {}.",
            num_phases - 1,
            op->debugName(),
            op->getExecutionPhase());
      }

      opInserter.sanitizePlacementAnnotation(
          op, has_phase ? op->getExecutionPhase() : phase);
    }

    // Recomputation annotation
    logging::transform::debug(
        "[StreamingMemory] Recomputation & Tensor Location annotation");
    for (auto &op : graph.getOps()) {
      // Mark any random seed operators as OnChip.
      if (op.second->settings.tensorLocation.storage ==
          TensorStorage::Undefined) {
        if (op.second->opid == Onnx::CustomOperators::GetRandomSeed) {
          op.second->settings.tensorLocation.storage = TensorStorage::OnChip;
          logging::transform::trace("[StreamingMemory] {} set to OnChip",
                                    op.second->debugName());
        }
      }
      // Recompute everything else (fwd) not set by the user by default
      if (op.second->settings.recomputeType == RecomputeType::Undefined) {
        if (ir.autoRecomputationEnabled() && !op.second->isIpuCopyOp() &&
            !dynamic_cast<GetRandomSeedOp *>(op.second.get())) {
          op.second->settings.recomputeType = RecomputeType::Recompute;
          logging::transform::trace("[StreamingMemory] {} set to Recompute",
                                    op.second->debugName());
        } else {
          op.second->settings.recomputeType = RecomputeType::Checkpoint;
        }
      }
    }
  }

  // Figure out the right phase for ops that did not get a phase yet
  while (true) {
    int num_ops_without_phase = 0;
    // Need to get the schedule every time,
    // because setting phases can change schedule order
    for (Op *op : schedule) {
      if (!op->hasExecutionPhase()) {
        // Check which phase the consumers of the output are in
        op->inheritPlacementAttributes(true);

        if (op->hasExecutionPhase()) {
          // Make sure phase adheres to producer/consumer and topological
          // constraints
          opInserter.sanitizePlacementAnnotation(op, op->getExecutionPhase());
        } else {
          ++num_ops_without_phase;
        }
      }
    }
    if (num_ops_without_phase == 0) {
      break;
    }
  }

  if (pass == 2) {
    opInserter.apply();
  }

  verifyExecutionPhases(graph);
  return true;
}

namespace {
// StreamingMemory 1: Map ops to phases, enable caching on variables
bool init1 = Transform::registerTransform(new StreamingMemory(1));
// StreamingMemory 2: Enable caching on variables, map remaining ops to phases,
// cut graph and insert cache ops.
bool init2 = Transform::registerTransform(new StreamingMemory(2));
} // namespace

} // namespace popart
