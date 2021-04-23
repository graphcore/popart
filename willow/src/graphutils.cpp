// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <queue>

#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/op.hpp>
#include <popart/op/init.hpp>
#include <popart/op/subgraph.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

namespace popart {
namespace graphutils {

void traverse(std::vector<Tensor *> tensors,
              std::function<bool(Tensor *)> visitor,
              std::function<bool(Op *, Tensor *, Tensor *)> filter,
              TraversalType traversalType,
              VisitType visitType,
              TraversalDirection traversalDirection) {
  std::deque<Tensor *> deque;
  std::set<Tensor *> visited;

  auto enqueue = [&deque, &visited](Tensor *t) {
    if (visited.find(t) == visited.end()) {
      visited.insert(t);
      deque.push_back(t);
    }
  };

  for (Tensor *t : tensors) {
    enqueue(t);
  }

  while (!deque.empty()) {
    Tensor *tq;

    switch (traversalType) {
    case TraversalType::BreadthFirst: {
      tq = deque.front();
      deque.pop_front();
      break;
    }
    case TraversalType::DepthFirst: {
      tq = deque.back();
      deque.pop_back();
      break;
    }
    }

    bool keep_going = false;

    if (visitType == VisitType::Pre) {
      keep_going = visitor(tq);
    }

    std::vector<Tensor *> toEnqueue;

    auto addOpBwd = [&toEnqueue, &tq, &filter](Op *op) {
      for (auto &input : op->input->tensorMap()) {
        Tensor *tn = input.second;
        if (filter(op, tq, tn)) {
          toEnqueue.push_back(tn);
        }
      }
    };

    auto addOpFwd = [&toEnqueue, &tq, &filter](Op *op) {
      for (auto &output : op->output->tensorMap()) {
        Tensor *tn = output.second;
        if (filter(op, tq, tn)) {
          toEnqueue.push_back(tn);
        }
      }
    };

    auto bwd = [&tq, &addOpBwd, &toEnqueue, &filter]() {
      // Producer Ops
      if (tq->hasProducer()) {
        Op *p = tq->getProducer();
        if (SubgraphOp *sgOp = dynamic_cast<SubgraphOp *>(p)) {
          // Subgraph Op
          auto indices = sgOp->output->indices(tq);
          for (OutIndex opOutIndex : indices) {
            OutIndex sgOutIndex = sgOp->opOutToSubgraphOutIndex(opOutIndex);
            if (sgOutIndex >= 0 &&
                sgOutIndex < sgOp->getCalledGraph().getOutputIds().size()) {
              TensorId sgOutId = sgOp->getCalledGraph().getOutputId(sgOutIndex);
              Tensor *sgOut = sgOp->getCalledGraph().getTensors().get(sgOutId);
              if (filter(sgOp, tq, sgOut)) {
                toEnqueue.push_back(sgOut);
              }
            }
          }
        } else {
          // Regular Op
          addOpBwd(p);
        }
      }

      // Graph inputs
      if (tq->isGraphInput()) {
        auto callSites = tq->getGraph().getCallSiteOps();
        for (Op *callSite : callSites) {
          if (SubgraphOp *sgOp = dynamic_cast<SubgraphOp *>(callSite)) {
            InIndex index =
                sgOp->subgraphInToOpInIndex(tq->getGraphInputIndex());
            if (sgOp->hasInput(index) &&
                filter(sgOp, tq, sgOp->inTensor(index))) {
              toEnqueue.push_back(sgOp->inTensor(index));
            }
          }
        }
      }
    };

    auto fwd = [&tq, &addOpFwd, &toEnqueue, &filter]() {
      // Consumer Ops
      for (Op *c : tq->consumers.getOps()) {
        if (SubgraphOp *sgOp = dynamic_cast<SubgraphOp *>(c)) {
          // Subgraph Op
          auto indices = sgOp->input->indices(tq);
          for (InIndex opInIndex : indices) {
            InIndex sgInIndex = sgOp->opInToSubgraphInIndex(opInIndex);
            if (sgInIndex >= 0 &&
                sgInIndex < sgOp->getCalledGraph().getInputIds().size()) {
              TensorId sgInId = sgOp->getCalledGraph().getInputId(sgInIndex);
              Tensor *sgIn    = sgOp->getCalledGraph().getTensors().get(sgInId);
              if (filter(sgOp, tq, sgIn)) {
                toEnqueue.push_back(sgIn);
              }
            }
          }
        } else {
          // Regular Op
          addOpFwd(c);
        }
      }

      // Graph outputs
      if (tq->isGraphOutput()) {
        auto callSites = tq->getGraph().getCallSiteOps();
        for (Op *callSite : callSites) {
          if (SubgraphOp *sgOp = dynamic_cast<SubgraphOp *>(callSite)) {
            InIndex index =
                sgOp->subgraphOutToOpOutIndex(tq->getGraphOutputIndex());
            if (sgOp->hasOutput(index) &&
                filter(sgOp, tq, sgOp->outTensor(index))) {
              toEnqueue.push_back(sgOp->outTensor(index));
            }
          }
        }
      }
    };

    switch (traversalDirection) {
    case TraversalDirection::ForwardBackward: {
      if (traversalType == TraversalType::BreadthFirst) {
        fwd();
        bwd();
      } else {
        bwd();
        fwd();
      }
      break;
    }
    case TraversalDirection::BackwardForward: {
      if (traversalType == TraversalType::BreadthFirst) {
        bwd();
        fwd();
      } else {
        fwd();
        bwd();
      }
      break;
    }
    case TraversalDirection::Forward: {
      fwd();
      break;
    }
    case TraversalDirection::Backward: {
      bwd();
      break;
    }
    }

    if (visitType == VisitType::Post) {
      keep_going = visitor(tq);
    }

    // Explore graph further along this tensor
    if (keep_going) {
      for (Tensor *te : toEnqueue) {
        enqueue(te);
      }
    }
  }
}

void traverseBreadthFirst(std::vector<Tensor *> tensors,
                          std::function<bool(Tensor *)> visitor) {
  auto filter = [](Op *op, Tensor *tq, Tensor *tn) { return true; };

  return traverse(tensors,
                  visitor,
                  filter,
                  TraversalType::BreadthFirst,
                  VisitType::Pre,
                  TraversalDirection::ForwardBackward);
}

void traverseDepthFirst(std::vector<Tensor *> tensors,
                        std::function<bool(Tensor *)> visitor) {
  auto filter = [](Op *op, Tensor *tq, Tensor *tn) { return true; };

  return traverse(tensors,
                  visitor,
                  filter,
                  TraversalType::DepthFirst,
                  VisitType::Pre,
                  TraversalDirection::ForwardBackward);
}

std::vector<Tensor *> rootTensors(Tensor *tensor) {
  std::vector<Tensor *> roots;
  auto visitor = [&roots](Tensor *t) {
    if ((!t->hasProducer() || t->getProducer()->isConvertibleTo<InitOp>()) &&
        !t->isGraphInput()) {
      roots.push_back(t);
    }
    return true;
  };

  auto filter = [](Op *op, Tensor *tq, Tensor *tn) { return true; };

  traverse({tensor},
           visitor,
           filter,
           TraversalType::DepthFirst,
           VisitType::Pre,
           TraversalDirection::Backward);
  return roots;
}

std::map<Op *, std::set<Op *>> getOpsWithBefores(const std::set<Op *> &ops) {
  std::map<Op *, std::set<Op *>> opsWithBefores;

  for (Op *op0 : ops) {
    opsWithBefores.insert({op0, {}});
  }

  if (ops.empty()) {
    return opsWithBefores;
  }

  Graph &graph = (*ops.begin())->getGraph();

  for (Op *op0 : ops) {
    // Graph connections
    std::queue<Op *> queue;
    queue.push(op0);
    while (!queue.empty()) {
      Op *op1 = queue.front();
      queue.pop();
      if (op1 != op0 && ops.find(op1) != ops.end()) {
        // Op1 occurs before Op0
        opsWithBefores[op0].insert(op1);
      }
      for (Tensor *input : op1->input->tensors()) {
        if (input->hasProducer()) {
          queue.push(input->getProducer());
        }
      }
    }

    // Topological constraints
    auto befores     = graph.topoCons->getBefores(op0);
    auto tiedBefores = graph.topoCons->getTiedBefores(op0);

    for (Op *op1 : befores) {
      if (ops.find(op1) != ops.end()) {
        opsWithBefores[op0].insert(op1);
      }
    }
    for (Op *op1 : tiedBefores) {
      if (ops.find(op1) != ops.end()) {
        opsWithBefores[op0].insert(op1);
      }
    }
  }

  // Transitive closure
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &befores : opsWithBefores) {
      for (auto &existingBefore : befores.second) {
        auto &newBefores = opsWithBefores.at(existingBefore);
        auto oldSize     = befores.second.size();
        befores.second.insert(newBefores.begin(), newBefores.end());
        changed |= befores.second.size() > oldSize;
      }
    }
  }

  return opsWithBefores;
}

std::map<Op *, std::set<Op *>> getOpsWithBefores(const std::vector<Op *> &ops) {
  std::set<Op *> opss;
  opss.insert(ops.begin(), ops.end());
  return getOpsWithBefores(opss);
}

} // namespace graphutils
} // namespace popart
