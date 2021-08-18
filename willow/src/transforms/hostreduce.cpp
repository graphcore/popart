// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/copyvarupdate.hpp>
#include <popart/op/hostreducevarupdate.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/sgd1acclupdate.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

#include <popart/transforms/hostreduce.hpp>

namespace popart {

std::size_t HostReduce::id() { return typeid(HostReduce).hash_code(); }

Op *HostReduce::insertGradCopyToHostOp(Op *varUpdateOp,
                                       Graph &graph,
                                       int counter) const {

  VarUpdateWithUpdaterOp *vop =
      dynamic_cast<VarUpdateWithUpdaterOp *>(varUpdateOp);

  // for the SGD1 case the gradTensor corresponds to velocity
  auto gradTensorId =
      vop->input->id(VarUpdateWithUpdaterOp::getUpdaterInIndex());

  std::string gradCopyOpName;
  if (vop->getName().empty()) {
    gradCopyOpName = hostReduceGradCopyPrefix() + std::string("d2h__") +
                     std::to_string(counter);
  } else {
    gradCopyOpName =
        hostReduceGradCopyPrefix() + std::string("d2h__") + vop->getName();
  }
  auto gradCopyOp_up =
      std::make_unique<GradCopyToHostOp>(Op::Settings(graph, gradCopyOpName));

  auto gradCopyOpId = graph.moveIntoGraph(std::move(gradCopyOp_up));
  auto gradCopyOp   = graph.getOp(gradCopyOpId);
  logging::transform::trace("Created GradCopyToHost operation: {}",
                            gradCopyOp->debugName());
  gradCopyOp->connectInTensor(GradCopyToHostOp::getInIndex(), gradTensorId);

  if (vop->hasVirtualGraphId()) {
    gradCopyOp->setVirtualGraphId(vop->getVirtualGraphId());
  }

  if (vop->hasPipelineStage()) {
    gradCopyOp->setPipelineStage(vop->getPipelineStage());
  }

  gradCopyOp->settings.schedulePriority = std::numeric_limits<double>::max();
  gradCopyOp->setup();

  if (gradCopyOp->getIr().getSessionOptions().enableGradientAccumulation) {
    gradCopyOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
  }

  return gradCopyOp;
}

Op *HostReduce::insertGradCopyFromHostOp(Op *varUpdateOp,
                                         Graph &graph,
                                         int counter) const {

  VarUpdateWithUpdaterOp *vop =
      dynamic_cast<VarUpdateWithUpdaterOp *>(varUpdateOp);

  auto gradTensorId =
      vop->input->id(VarUpdateWithUpdaterOp::getUpdaterInIndex());

  std::string gradCopyOpName;
  if (vop->getName().empty()) {
    gradCopyOpName = hostReduceGradCopyPrefix() + std::string("h2d__") +
                     std::to_string(counter);
  } else {
    gradCopyOpName =
        hostReduceGradCopyPrefix() + std::string("h2d__") + vop->getName();
  }
  auto gradCopyOp_up =
      std::make_unique<GradCopyFromHostOp>(Op::Settings(graph, gradCopyOpName));

  auto gradCopyOpId = graph.moveIntoGraph(std::move(gradCopyOp_up));
  auto gradCopyOp   = graph.getOp(gradCopyOpId);
  logging::transform::trace("Created GradCopyFromHost operation: {}",
                            gradCopyOp->debugName());
  gradCopyOp->connectInTensor(GradCopyFromHostOp::getInIndex(), gradTensorId);
  gradCopyOp->createAndConnectOutTensor(GradCopyFromHostOp::getOutIndex(),
                                        "HostReduceOut__" + gradTensorId);

  if (vop->hasVirtualGraphId()) {
    gradCopyOp->setVirtualGraphId(vop->getVirtualGraphId());
  }

  if (vop->hasPipelineStage()) {
    gradCopyOp->setPipelineStage(vop->getPipelineStage());
  }

  gradCopyOp->settings.schedulePriority = std::numeric_limits<double>::lowest();
  gradCopyOp->setup();

  if (gradCopyOp->getIr().getSessionOptions().enableGradientAccumulation) {
    gradCopyOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
  }

  return gradCopyOp;
}

Op *HostReduce::insertVarCopyOp(Op *varUpdateOp,
                                Graph &graph,
                                int counter) const {
  auto vop             = dynamic_cast<SGD0VarUpdateOp *>(varUpdateOp);
  auto optimizerInputs = vop->optimizerInputs();
  auto gradTensorId    = vop->input->id(SGD0VarUpdateOp::getUpdaterInIndex());
  auto varTensorId = vop->input->id(SGD0VarUpdateOp::getVarToUpdateInIndex());

  std::string varCopyOpName;
  if (vop->getName().empty()) {
    varCopyOpName = hostReduceVarCopyPrefix() + std::to_string(counter);
  } else {
    varCopyOpName = hostReduceVarCopyPrefix() + vop->getName();
  }
  auto varCopyOp_up = std::make_unique<HostSGD0VarUpdate>(
      vop->initSlr0, vop->initWdsf0, Op::Settings(graph, varCopyOpName));

  auto varCopyOpId = graph.moveIntoGraph(std::move(varCopyOp_up));
  auto varCopyOp   = graph.getOp(varCopyOpId);

  logging::transform::trace("Created HostSGD0VarUpdate operation: {}",
                            varCopyOp->debugName());
  varCopyOp->connectInTensor(SGD0VarUpdateOp::getUpdaterInIndex(),
                             gradTensorId);
  varCopyOp->connectInTensor(SGD0VarUpdateOp::getVarToUpdateInIndex(),
                             varTensorId);

  varCopyOp->createAndConnectOutTensor(SGD0VarUpdateOp::getUpdatedVarOutIndex(),
                                       "HostReduceOut__" + varTensorId);

  varCopyOp->setup();

  if (varUpdateOp->hasVirtualGraphId()) {
    varCopyOp->setVirtualGraphId(varUpdateOp->getVirtualGraphId());
  }
  for (auto &x : optimizerInputs) {
    varCopyOp->connectInTensor(x.first, x.second);
  }

  return varCopyOp;
}

void HostReduce::verifySessionOptions(const SessionOptions &options) const {
  if (!options.hostAllReduce) {
    return;
  }

  if (options.enablePipelining && options.hostWeightUpdate) {
    throw error("Pipelining with host weight update not supported");
  }

  if (options.enableGradientAccumulation && options.hostWeightUpdate) {
    throw error("Gradient accumulation with host weight update not supported");
  }

  if (options.hostAllReduceRemoteBuffer && options.hostWeightUpdate) {
    throw error("RemoteBuffer with host weight update not supported");
  }

  if (options.hostAllReduceRemoteBuffer && options.enableReplicatedGraphs) {
    throw error("RemoteBuffer with replicated graphs not supported");
  }
}

bool HostReduce::apply(Graph &graph) const {
  bool changed = false;
  auto &ir     = graph.getIr();

  verifySessionOptions(ir.getSessionOptions());

  logging::transform::debug("Applying HostReduce transformation");
  std::vector<Op *> gradCopyToHostOps;
  std::vector<Op *> gradCopyFromHostOps;
  std::vector<Op *> varCopyOps;

  std::vector<Op *> toRemove;

  int counter = 0;
  for (auto &id_op : graph.getOpSchedule({}, RequireOptimalSchedule::No)) {
    auto &op = id_op;

    const bool isVarUpdateOp = op->isConvertibleTo<SGD0VarUpdateOp>() ||
                               op->isConvertibleTo<SGD1VarUpdateOp>();
    if (!isVarUpdateOp) {
      continue;
    }
    if (op->isConvertibleTo<SGD1VarUpdateOp>() &&
        ir.getSessionOptions().hostWeightUpdate) {
      throw error("SGD1 is not supported with host weight update");
    }
    changed = true;

    auto gradCopyToHostOp = insertGradCopyToHostOp(op, graph, counter);
    gradCopyToHostOps.push_back(gradCopyToHostOp);

    if (ir.getSessionOptions().hostWeightUpdate) {
      auto varCopyOp = insertVarCopyOp(op, graph, counter);

      if (!varCopyOps.empty()) {
        graph.topoCons->insert(varCopyOps.back(), varCopyOp);
      }
      varCopyOps.push_back(varCopyOp);
      toRemove.push_back(op);

    } else {
      // device side weight update
      auto *gradCopyFromHostOp = insertGradCopyFromHostOp(op, graph, counter);
      gradCopyFromHostOps.push_back(gradCopyFromHostOp);

      const auto index         = VarUpdateWithUpdaterOp::getUpdaterInIndex();
      auto gradTensorId        = op->input->id(index);
      auto *originalGradTensor = ir.getTensor(gradTensorId);

      op->disconnectInTensor(index, originalGradTensor);
      op->connectInTensor(
          index, gradCopyFromHostOp->outId(GradCopyFromHostOp::getOutIndex()));
    }

    ++counter;
  }

  // A sync is added here to enforce that gradient copies to host are executed
  // before gradient/var copies to device. Gradient copies to host are scheduled
  // to happen before gradient/var copies to device in PopART. However, if
  // multiple stream copies are performed with a single sync id then a host
  // read can be scheduled before a host write in the Poplar engine but the
  // actual callback might still be executed after. This happens when Poplar
  // merges two host syncs during compilation into one. See
  // IPUTarget::prepareForStreamAccess() and IPUTarget::completeStreamAccess()
  // for details
  if (ir.getSessionOptions().enablePipelining ||
      ir.getSessionOptions().hostAllReduceRemoteBuffer) {
    if (gradCopyFromHostOps.size() != gradCopyToHostOps.size()) {
      throw error("Unequal number of gradient copy operations");
    }

    // Sync is added implicitly in opx for this case
    if (ir.getSessionOptions().enablePipelining) {
      for (const auto &gradCopyToHostOp : gradCopyToHostOps) {
        const auto toHostPs = gradCopyToHostOp->getPipelineStage();
        for (const auto &gradCopyFromHostOp : gradCopyFromHostOps) {
          const auto fromHostPs = gradCopyFromHostOp->getPipelineStage();
          if (toHostPs == fromHostPs) {
            graph.topoCons->insert(gradCopyToHostOp, gradCopyFromHostOp);
          }
        }
      }
    } else {
      for (auto &gradCopyFromHostOp : gradCopyFromHostOps) {
        graph.topoCons->insert({{gradCopyFromHostOp, gradCopyToHostOps}});
      }
    }
  } else {
    if (ir.getSessionOptions().hostWeightUpdate) {
      // Ensure that all gradient copy op run before var copy ops
      for (auto &varCopyOp : varCopyOps) {
        graph.topoCons->insert({{varCopyOp, gradCopyToHostOps}});
      }
    } else {
      // Ensure that all gradient copy from host run after gradient copy to host
      for (auto &gradCopyFromHostOp : gradCopyFromHostOps) {
        graph.topoCons->insert({{gradCopyFromHostOp, gradCopyToHostOps}});
      }
    }
  }

  const auto numOpsToRemove = toRemove.size();
  for (int i = 0; i < numOpsToRemove; ++i) {
    auto vop          = toRemove[i];
    auto outTensor    = vop->outTensor(VarUpdateOp::getUpdatedVarOutIndex());
    auto outTensorId  = outTensor->id;
    auto outTensorStr = outTensor->str();

    // disconnect and delete the single var updater and its output
    vop->disconnectAllInputs();
    vop->disconnectAllOutputs();
    graph.eraseOp(vop->id);
    graph.getTensors().remove(outTensorId);
  }

  return changed;
}

bool HostReduce::includesRequiredTensor(std::vector<const Tensor *> ts) {
  for (auto t : ts) {
    if (t->id.str().find(reservedGradientPrefix()) != std::string::npos) {
      for (auto con : t->consumers.getOps()) {
        if (con->isConvertibleTo<VarUpdateOp>() ||
            con->isConvertibleTo<GradCopyFromHostOp>() ||
            con->isConvertibleTo<GradCopyToHostOp>()) {
          return true;
        }
      }
    }
  }
  return false;
}

namespace {
bool init = Transform::registerTransform(new HostReduce);
}

} // namespace popart
