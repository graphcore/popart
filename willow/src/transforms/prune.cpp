// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

#include <popart/op/boundary.hpp>
#include <popart/op/init.hpp>
#include <popart/op/remote.hpp>
#include <popart/transforms/prune.hpp>

namespace popart {

std::size_t Prune::id() { return typeid(Prune).hash_code(); }

bool Prune::apply(Graph &graph) const {
  auto &ir = graph.getIr();

  // initialise with all the var
  // update ops for training,
  // and work backwards. This
  // is the set which is returned
  std::set<Op *> required = ir.getTrainTargetOps();

  // Find all ops which are not marked as pruneable and add those to required.
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (!op->pruneable) {
      required.insert(op);
    }
  }

  // as we work backwards, we keep a
  // "front" of tensors,
  std::vector<Tensor *> tensorFront;

  // when a tensor enters the "front",
  // we record that it has been visited
  std::set<Tensor *> tensorsVisited;

  // the "front" is initialsed with (1) anchor tensors,
  for (auto &tensorId : ir.getDataFlow().anchors()) {

    // Pruning can be run before anchors are validated.
    // There may be anchored tensors that aren't yet
    // present in the Ir.
    if (!graph.getTensors().contains(tensorId)) {
      continue;
    }

    Tensor *t = graph.getTensors().get(tensorId);
    // we have this check here as we allow
    // duplicated names from the (careless!) user
    if (tensorsVisited.count(t) == 0) {
      tensorFront.push_back(t);
      tensorsVisited.insert(t);
    }
  }

  // and (2), inputs to the training targets.
  for (auto &op : required) {
    for (auto t_inds : op->input->indicesMap()) {
      Tensor *t = t_inds.first;
      if (tensorsVisited.count(t) == 0) {
        tensorFront.push_back(t);
        tensorsVisited.insert(t);
      }
    }
  }

  // and (3), special case tensors that affect the model
  // even though they may not have a path to the loss.
  // This is the case for:
  //  - the random seed tensor
  if (ir.getSessionOptions().enableStochasticRounding) {
    auto seedId = GetRandomSeedOp::getUpdatedSeedTensorId();
    auto t      = graph.getTensors().get(seedId);
    tensorFront.push_back(t);
    tensorsVisited.insert(t);
  }
  // - RemoteStore inputs
  // RemoteStore have no outputs and no consumers thereafter,
  // so they will be pruned even though they do something necessary
  // (e.g. store an updated weight tensor).
  // This may no longer be necessary once we have host-tensor representations
  // in the IR instead, that can represent the output of a RemoteStore op
  // TODO: T17309
  for (Op *op : ir.getAllOps()) {
    if (RemoteStoreOp *store = dynamic_cast<RemoteStoreOp *>(op)) {
      for (auto &tensor : store->input->tensorMap()) {
        tensorFront.push_back(tensor.second);
        tensorsVisited.insert(tensor.second);
      }
    }
  }

  while (tensorFront.size() != 0) {
    Tensor *t = tensorFront.back();
    tensorFront.resize(tensorFront.size() - 1);
    if (t->hasProducer()) {
      std::set<Op *> newRequired = {};
      // Tensor t is on a target path. If any its
      // consumers modify it, they are required
      for (Op *consumer : t->consumers.getOps()) {
        // at any of the indices which op consumes t,
        // does it modify t?
        for (InIndex index : consumer->input->indices(t)) {
          auto modified = consumer->modifies(index);
          if (!std::all_of(modified.begin(),
                           modified.end(),
                           [](const view::Region &r) { return r.isEmpty(); })) {
            newRequired.insert(consumer);
          }
        }
      }
      // the producer of t is required
      newRequired.insert(t->getProducer());

      for (Op *op : newRequired) {
        if (required.count(op) == 0) {
          required.insert(op);
          for (auto t_inds : op->input->indicesMap()) {
            Tensor *t_in = t_inds.first;
            if (tensorsVisited.count(t_in) == 0) {
              tensorFront.push_back(t_in);
              tensorsVisited.insert(t_in);
            }
          }
        }
      }
    }
  }

  // at this point, "required" is the set
  // of all ops which are actually executed
  // to get targets

  // ops \ required
  std::vector<Op *> opsToDelete;
  // all outputs of opsToDelete
  std::vector<Tensor *> tensorsToDelete;

  for (auto &id_op : graph.getOps()) {
    Op *op = id_op.second.get();
    // TODO: Better mechanism to preserve special ops
    if (required.count(op) == 0 && !dynamic_cast<RemoteStoreOp *>(op) &&
        !dynamic_cast<BoundaryOp *>(op)) {
      opsToDelete.push_back(op);
      for (auto &t_inds : op->output->indicesMap()) {
        tensorsToDelete.push_back(t_inds.first);
      }
    }
  }

  for (Op *op : opsToDelete) {
    logging::transform::debug("Pruning {}", op->debugName());
    // unwire the inputs
    for (auto index_tensor : op->input->tensorMap()) {
      Tensor *tensor = index_tensor.second;
      tensor->consumers.decrement(op);
    }
    // remove the topo cons which might exist
    logging::transform::debug("[Prune] Pruning operator {}", op->opid);
    graph.topoCons->remove(op);
    graph.eraseOp(op->id);
  }

  for (Tensor *tensor : tensorsToDelete) {
    logging::transform::debug("[Prune] Pruning tensor {}", tensor->id);
    graph.getTensors().remove(tensor->id);
  }

  if (graph.getOps().size() == 0) {
    // The graph is empty, nothing to do. Error message depends on whether this
    // is the top-level graph.
    if (graph.id == graph.getIr().getMainGraph().id) {
      throw error(
          "All operations in the main graph were pruned, nothing to compute");
    } else {
      throw error("All operations in graph {} were pruned, nothing to compute",
                  graph.id.str());
    }
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new Prune);
}

} // namespace popart
