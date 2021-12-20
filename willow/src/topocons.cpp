// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/tensor.hpp>
#include <popart/topocons.hpp>

namespace popart {

OpsBeforeKey TopoCons::finalConsumerCons(const Tensor *tensor, Op *last) const {

  if (tensor->consumers.n(last) == 0) {
    throw error("Cannot set " + last->str() + " as last consumer of " +
                tensor->id + ", as it is not a consumer.");
  }

  OpsBeforeKey ops;
  ops[last] = {};
  ops[last].reserve(tensor->consumers.getMap().size());

  for (Op *op : tensor->consumers.getOps()) {
    if (op != last) {
      ops[last].push_back(op);
    }
  }
  return ops;
}

void TopoCons::insert(const OpsBeforeKey &ops, bool tied) {

  for (auto &after_befores : ops) {
    Op *after                        = after_befores.first;
    const std::vector<Op *> &befores = after_befores.second;
    for (auto &before : befores) {
      insert(before, after, tied);
    }
  }
}

namespace {

std::vector<Op *>
getValsOrEmpty(Op *key,
               const std::map<Op *, std::set<TopoOp>, POpCmp> &M,
               bool tiedOnly) {
  auto found = M.find(key);
  if (found == M.end()) {
    return {};
  }
  std::vector<Op *> vals;
  vals.reserve(found->second.size());
  for (TopoOp after : found->second) {
    if (!tiedOnly || after.tied)
      vals.push_back(after.op);
  }
  return vals;
}
} // namespace

std::vector<Op *> TopoCons::getAfters(Op *before) const {
  return getValsOrEmpty(before, valsAfter, false);
}

std::vector<Op *> TopoCons::getBefores(Op *after) const {
  return getValsOrEmpty(after, valsBefore, false);
}

std::vector<Op *> TopoCons::getTiedAfters(Op *before) const {
  return getValsOrEmpty(before, valsAfter, true);
}

std::vector<Op *> TopoCons::getTiedBefores(Op *after) const {
  return getValsOrEmpty(after, valsBefore, true);
}

void TopoCons::transfer(Op *beforeTransfer, Op *afterTransfer, bool removeOld) {
  transferToMultiple(beforeTransfer, {afterTransfer}, removeOld);
}

void TopoCons::transferToSubgraph(Op *replacementOp,
                                  std::map<Op *, std::vector<Op *>> opRemaps,
                                  bool removeOld) {
  Graph &graph = replacementOp->getGraph();

  auto opInRemaps = [&](Op *op) { return opRemaps.find(op) != opRemaps.end(); };

  std::set<std::pair<Op *, bool>, POpBoolCmp> befores;
  std::set<std::pair<Op *, bool>, POpBoolCmp> afters;

  for (auto &opRemap : opRemaps) {
    for (auto before : graph.topoCons->getBefores(opRemap.first)) {
      if (opInRemaps(before)) {
        // Internal topocon
        for (Op *opr0 : opRemap.second) {
          Graph &subgraph = opr0->getGraph();
          for (Op *opr1 : opRemaps.at(before)) {
            subgraph.topoCons->insert(opr1, opr0, false);
          }
        }
      } else {
        // External topocon
        befores.insert({before, false});
      }
    }
    for (auto after : graph.topoCons->getAfters(opRemap.first)) {
      if (opInRemaps(after)) {
        // Internal topocon
        for (Op *opr0 : opRemap.second) {
          Graph &subgraph = opr0->getGraph();
          for (Op *opr1 : opRemaps.at(after)) {
            subgraph.topoCons->insert(opr0, opr1, false);
          }
        }
      } else {
        // External topocon
        afters.insert({after, false});
      }
    }
    for (auto before : graph.topoCons->getTiedBefores(opRemap.first)) {
      if (opInRemaps(before)) {
        // Internal topocon
        for (Op *opr0 : opRemap.second) {
          Graph &subgraph = opr0->getGraph();
          for (Op *opr1 : opRemaps.at(before)) {
            subgraph.topoCons->insert(opr1, opr0, true);
          }
        }
      } else {
        // External topocon
        befores.insert({before, true});
      }
    }
    for (auto after : graph.topoCons->getTiedAfters(opRemap.first)) {
      if (opInRemaps(after)) {
        // Internal topocon
        for (Op *opr0 : opRemap.second) {
          Graph &subgraph = opr0->getGraph();
          for (Op *opr1 : opRemaps.at(after)) {
            subgraph.topoCons->insert(opr0, opr1, true);
          }
        }
      } else {
        // External topocon
        afters.insert({after, true});
      }
    }
  }

  if (removeOld) {
    // Remove the existing topocons
    for (auto &opRemap : opRemaps) {
      graph.topoCons->remove(opRemap.first);
    }
  }

  // Add the topoCons for the replacement Op (external topocons)
  for (auto before : befores) {
    graph.topoCons->insert(before.first, replacementOp, before.second);
  }
  for (auto after : afters) {
    graph.topoCons->insert(replacementOp, after.first, after.second);
  }
}

void TopoCons::transferToMultiple(Op *beforeTransfer,
                                  const std::vector<Op *> &afterTransfer,
                                  bool removeOld) {

  if (!getBefores(beforeTransfer).empty() ||
      !getAfters(beforeTransfer).empty()) {
    std::ostringstream oss;
    oss << ' ';
    for (auto x : afterTransfer) {
      oss << x->str() << ' ';
    }

    logging::ir::debug("Transfering topological constraints from {} to [{}]",
                       beforeTransfer->str(),
                       oss.str());
  }

  // for all b : b -> beforeTransfer, insert
  //             b -> x for all x in afterTransfer. The edge
  //             b -> beforeTransfer will be removed at the end
  if (valsBefore.find(beforeTransfer) != valsBefore.end()) {
    for (TopoOp b : valsBefore[beforeTransfer]) {
      for (auto x : afterTransfer) {
        insert(b.op, x, b.tied);
      }
    }
  }

  // for all a : beforeTransfer -> a, insert
  //             afterTransfer -> a. The edge
  //             beforeTransfer -> a will be removed at the end
  if (valsAfter.find(beforeTransfer) != valsAfter.end()) {
    for (TopoOp a : valsAfter[beforeTransfer]) {
      for (auto x : afterTransfer) {
        insert(x, a.op, a.tied);
      }
    }
  }

  if (removeOld) {
    remove(beforeTransfer);
  }
}

bool TopoCons::contains(Op *before, Op *after) const {
  auto foundBefore = valsAfter.find(before);
  if (foundBefore != valsAfter.end()) {
    auto &afters = foundBefore->second;
    if (afters.find(after) != afters.end()) {
      return true;
    }
  }
  return false;
}

void TopoCons::remove(Op *op) {

  logging::ir::debug("Removing topological constraints from {}", op->str());

  for (Op *before : getBefores(op)) {
    valsAfter[before].erase(op);
  }
  for (Op *after : getAfters(op)) {
    valsBefore[after].erase(op);
  }
  valsBefore.erase(op);
  valsAfter.erase(op);
}

void TopoCons::remove(Op *before, Op *after) {
  valsAfter[before].erase(after);
  valsBefore[after].erase(before);
}

// insert the topological constraint before -> after
void TopoCons::insert(Op *before, Op *after, bool tied) {

  if (before == after) {
    throw error("Cannot have \"a -> a\" topological constraint");
  }

  // check that there is no edge "after -> before"
  auto found = valsAfter.find(after);
  if (found != valsAfter.end()) {
    if (found->second.find(before) != found->second.end()) {
      throw error(
          "Constraint \"{}->{}\" already present, cannot add \"{}->{}\"",
          after->str(),
          before->str(),
          before->str(),
          after->str());
    }
  }

  TopoOp topoAfter(after, tied);
  TopoOp topoBefore(before, tied);

  found = valsAfter.find(before);
  if (found != valsAfter.end()) {
    found->second.insert(topoAfter);
  } else {
    valsAfter[before] = {topoAfter};
  }

  found = valsBefore.find(after);
  if (found != valsBefore.end()) {
    found->second.insert(topoBefore);
  } else {
    valsBefore[after] = {topoBefore};
  }
}

bool TopoCons::hasConstraint(Op *op) {

  if ((valsBefore.find(op) == valsBefore.end()) &&
      ((valsAfter.find(op) == valsAfter.end()))) {
    return false;
  }

  return true;
}

std::ostream &operator<<(std::ostream &os, const TopoCons &tc) {
  os << "TopoCons:\n";

  os << "  valsAfter:\n";
  for (auto &op_opsAfter : tc.valsAfter) {
    auto op        = op_opsAfter.first;
    auto &opsAfter = op_opsAfter.second;

    os << logging::format("    {}:\n", op->debugName());
    for (auto o : opsAfter) {
      os << logging::format("      {}\n", o.op->debugName());
    }
  }

  os << "  valsBefore:\n";
  for (auto &op_opsBefore : tc.valsBefore) {
    auto op         = op_opsBefore.first;
    auto &opsBefore = op_opsBefore.second;

    os << logging::format("    {}:\n", op->debugName());
    for (auto o : opsBefore) {
      os << logging::format("      {}\n", o.op->debugName());
    }
  }

  return os;
}

} // namespace popart
