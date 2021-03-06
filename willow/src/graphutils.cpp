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
  std::vector<TensorAndCallStack> tensorsAndStack;
  tensorsAndStack.reserve(tensors.size());
  for (Tensor *t : tensors) {
    tensorsAndStack.push_back({t, {}});
  }
  traverse(tensorsAndStack,
           visitor,
           filter,
           traversalType,
           visitType,
           traversalDirection);
}

void traverse(std::vector<TensorAndCallStack> tensors,
              std::function<bool(Tensor *)> visitor,
              std::function<bool(Op *, Tensor *, Tensor *)> filter,
              TraversalType traversalType,
              VisitType visitType,
              TraversalDirection traversalDirection) {
  std::deque<TensorAndCallStack> deque;
  std::map<Tensor *, std::set<CallStack>> visited;

  auto enqueue = [&deque, &visited](TensorAndCallStack &t) {
    if (visited.find(t.first) == visited.end() ||
        (visited[t.first].find({}) == visited[t.first].end() &&
         visited[t.first].find(t.second) == visited[t.first].end())) {
      // No generic (no call stack) or call-stack specific visit of t has
      // occured
      visited[t.first].insert(t.second);
      deque.push_back(t);
    }
  };

  for (auto &t : tensors) {
    enqueue(t);
  }

  while (!deque.empty()) {
    TensorAndCallStack tq;

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
      keep_going = visitor(tq.first);
    }

    std::vector<TensorAndCallStack> toEnqueue;

    auto addOpBwd = [&toEnqueue, &tq, &filter](Op *op, CallStack stack) {
      for (auto &input : op->input->tensorMap()) {
        Tensor *tn = input.second;
        if (filter(op, tq.first, tn)) {
          toEnqueue.push_back({tn, stack});
        }
      }
    };

    auto addOpFwd = [&toEnqueue, &tq, &filter](Op *op, CallStack stack) {
      for (auto &output : op->output->tensorMap()) {
        Tensor *tn = output.second;
        if (filter(op, tq.first, tn)) {
          toEnqueue.push_back({tn, stack});
        }
      }
    };

    auto bwd = [&tq, &addOpBwd, &toEnqueue, &filter]() {
      // Producer Ops
      if (tq.first->hasProducer()) {
        Op *p = tq.first->getProducer();
        if (!p->getCalledGraphIds().empty()) {
          // Subgraph Op
          auto indices = p->output->indices(tq.first);
          auto stack   = tq.second;
          stack.push_back(p);
          for (OutIndex opOutIndex : indices) {
            for (auto pgraph : p->getCalledGraphs()) {
              OutIndex sgOutIndex = p->opOutToSubgraphOutIndex(
                  p->getCalledGraphIndex(pgraph->id), opOutIndex);
              if (sgOutIndex >= 0 &&
                  sgOutIndex < pgraph->getOutputIds().size()) {
                TensorId sgOutId = pgraph->getOutputId(sgOutIndex);
                Tensor *sgOut    = pgraph->getTensors().get(sgOutId);
                if (filter(p, tq.first, sgOut)) {
                  toEnqueue.push_back({sgOut, stack});
                }
              }
            }
          }
        } else {
          // Regular Op
          addOpBwd(p, tq.second);
        }
      }

      // Graph inputs
      if (tq.first->isGraphInput()) {
        auto processCallSite = [&toEnqueue, &filter, &tq](Op *sgOp) {
          InIndex index = sgOp->subgraphInToOpInIndex(
              sgOp->getCalledGraphIndex(tq.first->getGraph().id),
              tq.first->getGraphInputIndex());
          if (sgOp->hasInput(index) &&
              filter(sgOp, tq.first, sgOp->inTensor(index))) {
            auto stack = tq.second;
            if (!stack.empty() && stack.back() == sgOp) {
              stack.pop_back();
            }
            toEnqueue.push_back({sgOp->inTensor(index), stack});
          }
        };

        if (tq.second.empty()) {
          auto callSites = tq.first->getGraph().getCallSiteOps();
          for (Op *sgOp : callSites) {
            processCallSite(sgOp);
          }
        } else {
          processCallSite(tq.second.back());
        }
      }
    };

    auto fwd = [&tq, &addOpFwd, &toEnqueue, &filter]() {
      // Consumer Ops
      for (Op *c : tq.first->consumers.getOps()) {
        if (!c->getCalledGraphIds().empty()) {
          // Subgraph Op
          auto indices = c->input->indices(tq.first);
          auto stack   = tq.second;
          stack.push_back(c);
          for (InIndex opInIndex : indices) {
            for (auto cgraph : c->getCalledGraphs()) {
              InIndex sgInIndex = c->opInToSubgraphInIndex(
                  c->getCalledGraphIndex(cgraph->id), opInIndex);
              if (sgInIndex >= 0 && sgInIndex < cgraph->getInputIds().size()) {
                TensorId sgInId = cgraph->getInputId(sgInIndex);
                Tensor *sgIn    = cgraph->getTensors().get(sgInId);
                if (filter(c, tq.first, sgIn)) {
                  toEnqueue.push_back({sgIn, stack});
                }
              }
            }
          }
        } else {
          // Regular Op
          addOpFwd(c, tq.second);
        }
      }

      // Graph outputs
      if (tq.first->isGraphOutput()) {
        auto processCallSite = [&toEnqueue, &filter, &tq](Op *sgOp) {
          InIndex index = sgOp->subgraphOutToOpOutIndex(
              sgOp->getCalledGraphIndex(tq.first->getGraph().id),
              tq.first->getGraphOutputIndex());
          if (sgOp->hasOutput(index) &&
              filter(sgOp, tq.first, sgOp->outTensor(index))) {
            auto stack = tq.second;
            if (!stack.empty() && stack.back() == sgOp) {
              stack.pop_back();
            }
            toEnqueue.push_back({sgOp->outTensor(index), stack});
          }
        };

        if (tq.second.empty()) {
          // Exit through all call sites
          auto callSites = tq.first->getGraph().getCallSiteOps();
          for (Op *sgOp : callSites) {
            processCallSite(sgOp);
          }
        } else {
          // Exit through the current call stack call site
          processCallSite(tq.second.back());
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
      keep_going = visitor(tq.first);
    }

    // Explore graph further along this tensor
    if (keep_going) {
      for (TensorAndCallStack &te : toEnqueue) {
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

    graphutils::traverse(
        op0->input->tensors(),
        [&opsWithBefores, &op0, &ops](Tensor *t) {
          if (t->hasProducer()) {
            Op *op1 = t->getProducer();
            if (op0 != op1 && ops.find(op1) != ops.end()) {
              opsWithBefores[op0].insert(op1);
            }
          }
          return true;
        },
        [](Op *, Tensor *, Tensor *) { return true; },
        TraversalType::BreadthFirst,
        VisitType::Pre,
        TraversalDirection::Backward);

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
