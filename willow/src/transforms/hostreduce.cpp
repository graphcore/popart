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

bool HostReduce::apply(Graph &graph) const {
  bool changed = false;
  auto &ir     = graph.getIr();

  if (!ir.getSessionOptions().hostAllReduce)
    return changed;

  if (!ir.canTrain() && ir.getSessionOptions().hostAllReduce) {
    throw error("Host AllReduce only available when training.");
  }

  logging::transform::debug("Applying HostReduce transformation");
  std::vector<Op *> gradCopyOps;
  std::vector<Op *> varCopyOps;

  std::vector<Op *> toRemove;

  int counter = 0;
  for (auto &id_op : graph.getOps()) {
    auto &op = id_op.second;

    if (!op->isConvertibleTo<SGD0VarUpdateOp>()) {
      continue;
    }
    changed              = true;
    auto vop             = dynamic_cast<SGD0VarUpdateOp *>(op.get());
    auto optimizerInputs = vop->optimizerInputs();

    auto gradTensorId = vop->input->id(SGD0VarUpdateOp::getUpdaterInIndex());
    auto varTensorId = vop->input->id(SGD0VarUpdateOp::getVarToUpdateInIndex());

    std::string gradCopyOpName;
    if (vop->getName().empty()) {
      gradCopyOpName = hostReduceGradCopyPrefix() + std::to_string(counter);
    } else {
      gradCopyOpName = hostReduceGradCopyPrefix() + vop->getName();
    }
    auto gradCopyOp_up = std::make_unique<HostReduceGradCopyOp>(
        Op::Settings(graph, gradCopyOpName));

    auto gradCopyOpId = graph.moveIntoGraph(std::move(gradCopyOp_up));
    auto gradCopyOp   = graph.getOp(gradCopyOpId);
    logging::transform::trace("Created HostReduceGradCopy operation: {}",
                              gradCopyOp->debugName());
    gradCopyOp->connectInTensor(HostReduceGradCopyOp::getInIndex(),
                                gradTensorId);

    if (vop->hasVirtualGraphId()) {
      gradCopyOp->setVirtualGraphId(vop->getVirtualGraphId());
    }

    gradCopyOp->setup();

    graph.topoCons->transfer(vop, gradCopyOp);

    gradCopyOps.push_back(gradCopyOp);

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

    varCopyOp->createAndConnectOutTensor(
        SGD0VarUpdateOp::getUpdatedVarOutIndex(),
        "HostReduceOut__" + varTensorId);

    varCopyOp->setup();

    if (op->hasVirtualGraphId()) {
      varCopyOp->setVirtualGraphId(op->getVirtualGraphId());
    }

    varCopyOps.push_back(varCopyOp);

    for (auto &x : optimizerInputs) {
      gradCopyOp->connectInTensor(x.first, x.second);
      varCopyOp->connectInTensor(x.first, x.second);
    }

    toRemove.push_back(op.get());
    ++counter;
    logging::transform::debug(
        "HostReduce replaced {} with grad copy {} and var copy {}",
        vop->getName(),
        gradCopyOpName,
        varCopyOpName);
  }

  // Ensure that all gradient copy op run before var copy ops
  for (auto &op : varCopyOps) {
    graph.topoCons->insert({{op, gradCopyOps}});
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
