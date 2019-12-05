#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/copyvarupdate.hpp>
#include <popart/op/hostreducevarupdate.hpp>
#include <popart/op/sgd0varupdate.hpp>
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
  auto vop             = dynamic_cast<SGD0VarUpdateOp *>(varUpdateOp);
  auto optimizerInputs = vop->optimizerInputs();
  auto gradTensorId    = vop->input->id(SGD0VarUpdateOp::getUpdaterInIndex());

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

  gradCopyOp->priority = std::numeric_limits<double>::max();
  gradCopyOp->setup();

  graph.topoCons->transfer(vop, gradCopyOp);
  for (auto &x : optimizerInputs) {
    gradCopyOp->connectInTensor(x.first, x.second);
  }

  return gradCopyOp;
}

Op *HostReduce::insertGradCopyFromHostOp(Op *varUpdateOp,
                                         Graph &graph,
                                         int counter) const {
  auto vop             = dynamic_cast<SGD0VarUpdateOp *>(varUpdateOp);
  auto optimizerInputs = vop->optimizerInputs();
  auto gradTensorId    = vop->input->id(SGD0VarUpdateOp::getUpdaterInIndex());

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

  gradCopyOp->priority = std::numeric_limits<double>::min();
  gradCopyOp->setup();

  graph.topoCons->transfer(vop, gradCopyOp);
  for (auto &x : optimizerInputs) {
    gradCopyOp->connectInTensor(x.first, x.second);
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
  auto varCopyOp_up =
      std::make_unique<HostSGD0VarUpdate>(varTensorId,
                                          vop->initSlr0,
                                          vop->initWdsf0,
                                          Op::Settings(graph, varCopyOpName));

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

  varCopyOp->priority = std::numeric_limits<double>::min();
  varCopyOp->setup();

  if (varUpdateOp->hasVirtualGraphId()) {
    varCopyOp->setVirtualGraphId(varUpdateOp->getVirtualGraphId());
  }
  for (auto &x : optimizerInputs) {
    varCopyOp->connectInTensor(x.first, x.second);
  }

  return varCopyOp;
}

bool HostReduce::apply(Graph &graph) const {
  bool changed = false;
  auto &ir     = graph.getIr();

  if (!ir.getSessionOptions().hostAllReduce)
    return changed;

  if (!ir.canTrain() && ir.getSessionOptions().hostAllReduce) {
    throw error("Host AllReduce only available when training.");
  }

  logging::transform::debug("Applying HostReduce transformation");
  std::vector<Op *> gradCopyToHostOps;
  std::vector<Op *> gradCopyFromHostOps;
  std::vector<Op *> varCopyOps;

  std::vector<Op *> toRemove;

  int counter = 0;
  for (auto &id_op : graph.getOpSchedule({})) {
    auto &op = id_op;

    if (!op->isConvertibleTo<SGD0VarUpdateOp>()) {
      continue;
    }
    changed = true;

    auto gradCopyOp = insertGradCopyToHostOp(op, graph, counter);
    gradCopyToHostOps.push_back(gradCopyOp);

    if (ir.getSessionOptions().hostWeightUpdate) {
      auto varCopyOp = insertVarCopyOp(op, graph, counter);

      if (!varCopyOps.empty()) {
        graph.topoCons->insert(varCopyOps.back(), varCopyOp);
      }
      varCopyOps.push_back(varCopyOp);
      toRemove.push_back(op);

      logging::transform::debug(
          "HostReduce replaced {} with grad copy {} and var copy {}",
          op->getName(),
          gradCopyOp->getName(),
          varCopyOp->getName());
    } else {
      // device side weight update
      auto *gradCopyOp = insertGradCopyFromHostOp(op, graph, counter);
      gradCopyFromHostOps.push_back(gradCopyOp);

      const auto index         = SGD0VarUpdateOp::getUpdaterInIndex();
      auto gradTensorId        = op->input->id(index);
      auto *originalGradTensor = ir.getTensor(gradTensorId);

      op->disconnectInTensor(index, originalGradTensor);
      op->connectInTensor(index,
                          gradCopyOp->outId(GradCopyFromHostOp::getOutIndex()));
    }

    ++counter;
  }

  if (ir.getSessionOptions().hostWeightUpdate) {
    // Ensure that all gradient copy op run before var copy ops
    for (auto &op : varCopyOps) {
      graph.topoCons->insert({{op, gradCopyToHostOps}});
    }
  } else {
    // Ensure that all gradient copy from host run after gradient copy to host
    for (auto &op : gradCopyFromHostOps) {
      graph.topoCons->insert({{op, gradCopyToHostOps}});
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

namespace {
bool init = Transform::registerTransform(new HostReduce);
}

} // namespace popart
