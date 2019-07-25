#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

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
          if (!consumer->modifies(index).isEmpty()) {
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
    if (required.count(op) == 0) {
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
    graph.topoCons->remove(op);
    graph.eraseOp(op->id);
  }

  for (Tensor *tensor : tensorsToDelete) {
    graph.getTensors().remove(tensor->id);
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new Prune);
}

} // namespace popart
