// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <queue>

#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/init.hpp>
#include <popart/op/subgraph.hpp>
#include <popart/pointercomparators.hpp>
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
              TraversalDirection traversalDirection,
              TraverseCallSites traverseCallSites) {
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
           traversalDirection,
           traverseCallSites);
}

void traverse(std::vector<Tensor *> tensors,
              std::function<bool(Tensor *)> visitor,
              std::function<bool(Op *, Tensor *, Tensor *)> filter,
              TraversalType traversalType,
              VisitType visitType,
              TraversalDirection traversalDirection) {
  traverse(tensors,
           visitor,
           filter,
           traversalType,
           visitType,
           traversalDirection,
           TraverseCallSites::Current);
}

void traverse(std::vector<TensorAndCallStack> tensors,
              std::function<bool(Tensor *)> visitor,
              std::function<bool(Op *, Tensor *, Tensor *)> filter,
              TraversalType traversalType,
              VisitType visitType,
              TraversalDirection traversalDirection,
              TraverseCallSites traverseCallSites) {
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

    auto addOpBwd = [&toEnqueue, &tq, &filter](Op *op,
                                               TensorAndCallStack stack) {
      for (auto outIndex : op->output->indices(stack.first)) {
        for (auto inIndex : op->opOutToOpInIndex(outIndex)) {
          auto tn = op->inTensor(inIndex);
          if (filter(op, tq.first, tn)) {
            toEnqueue.push_back({tn, stack.second});
          }
        }
      }
    };

    auto addOpFwd = [&toEnqueue, &tq, &filter](Op *op,
                                               TensorAndCallStack stack) {
      for (auto inIndex : op->input->indices(stack.first)) {
        for (auto outIndex : op->opInToOpOutIndex(inIndex)) {
          auto tn = op->outTensor(outIndex);
          if (filter(op, tq.first, tn)) {
            toEnqueue.push_back({tn, stack.second});
          }
        }
      }
    };

    auto bwd = [&tq, &addOpBwd, &toEnqueue, &filter, &traverseCallSites]() {
      // Producer Ops
      if (tq.first->hasProducer()) {
        Op *p = tq.first->getProducer();
        if (!p->getCalledGraphIds().empty()) {
          // Subgraph Op
          auto indices = p->output->indices(tq.first);
          auto stack   = tq.second;
          if (traverseCallSites == TraverseCallSites::Current) {
            stack.push_back(p);
          }
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
        }
        addOpBwd(p, tq);
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

        if (traverseCallSites == TraverseCallSites::All || tq.second.empty()) {
          auto callSites = tq.first->getGraph().getCallSiteOps();
          for (Op *sgOp : callSites) {
            processCallSite(sgOp);
          }
        } else {
          processCallSite(tq.second.back());
        }
      }
    };

    auto fwd = [&tq, &addOpFwd, &toEnqueue, &filter, &traverseCallSites]() {
      // Consumer Ops
      for (Op *c : tq.first->consumers.getOps()) {
        if (!c->getCalledGraphIds().empty()) {
          // Subgraph Op
          auto indices = c->input->indices(tq.first);
          auto stack   = tq.second;
          if (traverseCallSites == TraverseCallSites::Current) {
            stack.push_back(c);
          }
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
        }
        addOpFwd(c, tq);
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

        if (traverseCallSites == TraverseCallSites::All || tq.second.empty()) {
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

std::map<Op *, std::set<Op *, POpCmp>, POpCmp>
getOpsWithBefores(const std::set<Op *, POpCmp> &ops) {
  std::map<Op *, std::set<Op *, POpCmp>, POpCmp> opsWithBefores;

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

std::map<Op *, std::set<Op *, POpCmp>, POpCmp>
getOpsWithBefores(const std::vector<Op *> &ops) {
  std::set<Op *, POpCmp> opss;
  opss.insert(ops.begin(), ops.end());
  return getOpsWithBefores(opss);
}

namespace {
class PartialMatch {
public:
  std::vector<Op *> finalize() {
    std::vector<Op *> opVec;
    for (auto &op : ops) {
      opVec.push_back(op.second);
    }
    return opVec;
  }

  // Ops that match the structure and predicate at index
  std::map<int, Op *> ops;

  // Number of Ops before the current predicate index can be processed
  std::map<int, int> befores;
};
} // namespace

bool operator<(const Edge &a, const Edge &b) {
  std::vector<int> av{a.getFrom(), a.getTo(), a.getOut(), a.getIn()};
  std::vector<int> bv{b.getFrom(), b.getTo(), b.getOut(), b.getIn()};
  return av < bv;
}

std::vector<std::vector<Op *>>
findMatchingOps(Graph &graph, OpPreds preds, Edges edges) {

  std::map<int, std::set<Edge>> outEdgeMap;
  std::map<int, std::set<Edge>> inEdgeMap;

  for (auto &edge : edges) {
    outEdgeMap[edge.getFrom()].insert(edge);
    inEdgeMap[edge.getTo()].insert(edge);
  }

  std::map<int, int> befores;
  for (int i = 0; i < preds.size(); ++i) {
    befores[i] = 0;
  }

  for (auto &edge : edges) {
    befores[edge.getTo()] += 1;
  }

  std::vector<PartialMatch> partialMatches;
  PartialMatch match;
  match.befores = befores;
  partialMatches.push_back(match);

  std::vector<std::vector<Op *>> matches;

  while (!partialMatches.empty()) {
    auto match = partialMatches.back();
    partialMatches.pop_back();

    int index = -1;
    for (auto &before : match.befores) {
      if (before.second == 0 &&
          match.ops.find(before.first) == match.ops.end()) {
        index = before.first;
      }
    }

    if (index != -1) {
      std::set<Op *, POpCmp> candidates;

      auto inEdges = inEdgeMap.find(index);

      if (inEdges == inEdgeMap.end() || inEdges->second.size() == 0) {
        auto &gOps = graph.getOps();
        for (auto &gOp : gOps) {
          if (preds.at(index)(gOp.second.get())) {
            candidates.insert(gOp.second.get());
          }
        }
      } else {
        bool first = true;
        for (auto &inEdge : inEdges->second) {
          auto fromOp = match.ops.at(inEdge.getFrom());
          std::set<Tensor *> tensors;
          if (inEdge.getOut() == -1) {
            auto fromTensors = fromOp->output->tensors();
            tensors.insert(fromTensors.begin(), fromTensors.end());
          } else {
            if (fromOp->hasOutput(inEdge.getOut())) {
              tensors.insert(fromOp->output->tensor(inEdge.getOut()));
            }
          }
          std::set<Op *, POpCmp> localCandidates;
          for (auto t : tensors) {
            for (auto c : t->consumers.getOps()) {
              if (((inEdge.getIn() == -1) ||
                   ((c->input->hasIndex(inEdge.getIn())) &&
                    (c->input->tensor(inEdge.getIn()) == t))) &&
                  preds.at(index)(c)) {
                localCandidates.insert(c);
              }
            }
          }
          if (first) {
            candidates = localCandidates;
          } else {
            std::set<Op *, POpCmp> intersectCandidates;
            std::set_intersection(
                candidates.begin(),
                candidates.end(),
                localCandidates.begin(),
                localCandidates.end(),
                std::inserter(intersectCandidates, intersectCandidates.begin()),
                [](const Op *a, const Op *b) { return a->id < b->id; });
            candidates = intersectCandidates;
          }
          first = false;
        }
      }

      for (auto candidate : candidates) {
        auto nextMatch       = match;
        nextMatch.ops[index] = candidate;
        auto outEdges        = outEdgeMap.find(index);
        if (outEdges != outEdgeMap.end()) {
          for (auto &edge : outEdges->second) {
            nextMatch.befores.at(edge.getTo())--;
          }
        }
        if (nextMatch.ops.size() == preds.size()) {
          matches.push_back(nextMatch.finalize());
        } else {
          partialMatches.push_back(nextMatch);
        }
      }
    }
  }

  return matches;
}

} // namespace graphutils
} // namespace popart
