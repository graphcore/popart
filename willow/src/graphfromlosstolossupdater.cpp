// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {
namespace graphFromLossToLossUpdater {

namespace {

// move backwards through the inputs and their producers
std::set<Vertex *> backwardPropogate(std::vector<Op *> frontier) {
  std::set<Vertex *> visited;
  for (auto x : frontier) {
    visited.emplace(x);
  }
  while (frontier.size() > 0) {
    auto toProcess = frontier.back();
    frontier.resize(frontier.size() - 1);
    // get all producers of inputs, add them to the frontier
    for (auto inTensor : toProcess->input->tensors()) {
      visited.emplace(inTensor);
      auto producer = inTensor->getProducerUnsafe();
      if (producer && visited.count(producer) == 0) {
        visited.emplace(producer);
        frontier.push_back(producer);
      }
    }
  }
  return visited;
}

// move forwards the the outputs and their consumers
std::set<Vertex *> forwardPropogate(std::vector<Op *> frontier) {
  std::set<Vertex *> visited;
  for (auto x : frontier) {
    visited.emplace(x);
  }
  while (frontier.size() > 0) {
    auto toProcess = frontier.back();
    frontier.resize(frontier.size() - 1);
    for (auto outTensor : toProcess->output->tensors()) {
      visited.emplace(outTensor);
      for (auto consumer : outTensor->consumers.getOps()) {
        if (visited.count(consumer) == 0) {
          visited.emplace(consumer);
          frontier.push_back(consumer);
        }
      }
    }
  }
  return visited;
}

} // namespace

void propagate(Graph &g) {
  // 1) Get all Ops which have toLoss Yes, and backwards propagate
  std::vector<Op *> toLossFrontier;
  // 2) Get all Ops which have fromLoss Yes, and forwards propagate
  std::vector<Op *> fromLossFrontier;

  for (auto &id_op : g.getOps()) {
    Op *op = id_op.second.get();
    if (op->toLoss == PathToLoss::Yes) {
      toLossFrontier.push_back(op);
    }

    if (op->fromLoss == PathFromLoss::Yes) {
      fromLossFrontier.push_back(op);
    }

    // If an Op's input has PathFromLoss::Yes, then so do does Op
    for (auto arr : op->input->tensors()) {
      if (arr->fromLoss == PathFromLoss::Yes) {
        op->fromLoss = PathFromLoss::Yes;
        fromLossFrontier.push_back(op);
      }
    }

    // If an Op's output has PathToLoss::Yes, then so do does Op
    for (auto arr : op->output->tensors()) {
      if (arr->toLoss == PathToLoss::Yes) {
        op->toLoss = PathToLoss::Yes;
        toLossFrontier.push_back(op);
      }
    }
  }

  auto toLossVertices = backwardPropogate(toLossFrontier);
  for (Vertex *v : toLossVertices) {
    v->toLoss = PathToLoss::Yes;
  }

  auto fromLossVertices = forwardPropogate(fromLossFrontier);
  for (Vertex *v : fromLossVertices) {
    v->fromLoss = PathFromLoss::Yes;
  }

  // set all Undefined to No
  for (auto &id_op : g.getOps()) {
    auto setUnPaths = [](Vertex *v) {
      if (v->toLoss == PathToLoss::Undefined) {
        v->toLoss = PathToLoss::No;
      }
      if (v->fromLoss == PathFromLoss::Undefined) {
        v->fromLoss = PathFromLoss::No;
      }
    };

    auto op = id_op.second.get();
    setUnPaths(op);
    for (auto tensor : op->input->tensors()) {
      setUnPaths(tensor);
    }
    for (auto tensor : op->output->tensors()) {
      setUnPaths(tensor);
    }
  }
}

void unsetAll(Graph &g) {
  for (auto &t : g.getTensors().getAll()) {
    t->fromLoss = PathFromLoss::Undefined;
    t->toLoss   = PathToLoss::Undefined;
  }
  for (auto &id_op : g.getOps()) {
    auto op      = id_op.second.get();
    op->fromLoss = PathFromLoss::Undefined;
    op->toLoss   = PathToLoss::Undefined;
  }
}

} // namespace graphFromLossToLossUpdater
} // namespace popart
