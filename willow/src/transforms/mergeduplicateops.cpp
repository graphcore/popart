// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/tensorindex.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/mergeduplicateops.hpp>

namespace popart {

using EquivId = std::string;

namespace {

EquivId getEquivId(const Op *op) {
  EquivId equivId = "Inputs[";
  for (auto &idx_id : op->input->tensorIdMap()) {
    auto idx = idx_id.first;
    auto id  = idx_id.second;
    equivId += logging::format("{}: {}, ", idx, id);
  }
  equivId += "],";
  equivId += op->getSubgraphEquivId();
  return equivId;
}

std::map<EquivId, std::vector<Op *>> getConsumerIdMap(const Tensor *tensor) {
  std::map<EquivId, std::vector<Op *>> equivIds;
  for (auto consumer : tensor->consumers.getOps()) {
    auto equivId = getEquivId(consumer);
    equivIds[equivId].push_back(consumer);
  }

  return equivIds;
}

void reconnectConsumerInput(Op *consumer, Tensor *from, Tensor *to) {
  auto indices = consumer->input->indices(from);

  for (auto i : indices) {
    if (auto copyOp = dynamic_cast<IpuCopyOp *>(consumer)) {
      auto source = copyOp->getSourceIpu(from->id);
      copyOp->disconnectInTensor(i, from);
      copyOp->connectInTensor(i, to->id, source);
    } else {
      consumer->disconnectInTensor(i, from);
      consumer->connectInTensor(i, to->id);
    }
  }
}

void reconnectConsumers(Tensor *from, Tensor *to) {
  auto consumers = from->consumers.getOps();
  for (auto c : consumers) {
    reconnectConsumerInput(c, from, to);
  }
}

// Reconnect the consumers of the outputs of `from` to consume the outputs of
// `to`.
void reconnectOutputConsumers(Op *from, Op *to) {
  for (auto &idx_tensor : from->output->tensorMap()) {
    auto idx        = idx_tensor.first;
    auto fromTensor = idx_tensor.second;
    auto toTensor   = to->outTensor(idx);
    reconnectConsumers(fromTensor, toTensor);
  }
}

void removeOpAndOutputs(Op *op) {
  auto &graph  = op->getGraph();
  auto outputs = op->output->tensors();
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  graph.eraseOp(op->id);
  for (auto out : outputs) {
    if (out->consumers.getTotal() > 0) {
      throw internal_error("All the consumers of tensor {} should have had a "
                           "replacement input connected already.",
                           out->str());
    }
    graph.getTensors().remove(out->id);
  }
}

bool outputsAnchor(const Op *op) {
  for (auto t : op->output->tensors()) {
    if (op->getIr().isAnchored(t->id)) {
      return true;
    }
  }
  return false;
}

bool hasTopoCons(Op *op) {
  auto &graph = op->getGraph();
  return graph.topoCons->hasConstraint(op);
}

void mergeDuplicateConsumers(const std::vector<Op *> &consumers) {
  std::stringstream ss;
  ss << "Attempting to merge consumers:";
  for (auto c : consumers) {
    ss << logging::format("\n  {}", c->debugName());
  }
  logging::transform::debug("{}", ss.str());

  // Take the first consumer as the master consumer
  auto master = consumers.at(0);

  std::vector<Op *> mergeableConsumers;
  for (int i = 1; i < consumers.size(); i++) {
    auto x = consumers.at(i);
    if (!outputsAnchor(x) && !hasTopoCons(x)) {
      mergeableConsumers.push_back(x);
    }
  }

  for (auto x : mergeableConsumers) {
    logging::transform::trace(
        "Merging\n  {}\ninto\n  {}", x->debugName(), master->debugName());
    reconnectOutputConsumers(x, master);
    removeOpAndOutputs(x);
  }
}

std::vector<TensorId> getAllOutputIds(const std::vector<Op *> &consumers) {
  std::vector<TensorId> outputs;

  for (auto c : consumers) {
    for (auto t : c->output->tensors()) {
      outputs.push_back(t->id);
    }
  }

  return outputs;
}

} // namespace

std::size_t MergeDuplicateOps::id() {
  return typeid(MergeDuplicateOps).hash_code();
}

bool MergeDuplicateOps::apply(Graph &graph) const {

  TensorSearchHelper frontier;

  // Populate pending with all tensors that don't have a producer.
  for (auto tensorId : graph.getTensors().getNoProducerIds()) {
    auto tensor = graph.getTensors().get(tensorId);
    frontier.push(tensor);
  }

  // Only push a tensor to frontier if all the inputs of the producing op have
  // already been popped.
  std::set<Tensor *> popped;
  auto tryPushToFrontier = [&](Tensor *t) {
    auto producer = t->getProducer();

    // An op should only be added to the frontier if all the producers inputs
    // have been processed already.
    for (auto input : producer->input->tensors()) {
      if (popped.find(input) == popped.end()) {
        return;
      }
    }

    frontier.push(t);
  };

  // Traverse through the graph, merging consumers where possible.
  // We traverse through the graph as changes earlier in the graph will affect
  // which consumers may be merged.
  while (!frontier.empty()) {
    auto tensor = frontier.pop();
    popped.insert(tensor);

    auto consumerIdMap = getConsumerIdMap(tensor);
    for (auto &id_consumers : consumerIdMap) {
      auto &consumers = id_consumers.second;

      auto recomputeEnabled =
          std::find_if(std::begin(consumers), std::end(consumers), [](Op *op) {
            return op->settings.recomputeType == RecomputeType::Recomputed;
          });

      if (consumers.size() > 1 && (recomputeEnabled == std::end(consumers))) {
        auto outIds = getAllOutputIds(consumers);
        mergeDuplicateConsumers(consumers);
        // Add any outputs that weren't erased to the frontier
        for (auto outId : outIds) {
          if (graph.getTensors().contains(outId)) {
            tryPushToFrontier(graph.getTensors().get(outId));
          }
        }
      } else {
        for (auto c : consumers) {
          for (auto o : c->output->tensors()) {
            tryPushToFrontier(o);
          }
        }
      }
    }
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new MergeDuplicateOps);
}

} // namespace popart
