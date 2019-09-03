#include <popart/op.hpp>
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

void TopoCons::insert(const OpsBeforeKey &ops) {

  for (auto &entry : ops) {
    logging::ir::debug("Inserting topological constraints from {} to {}",
                       entry.first->str(),
                       entry.second);
  }

  for (auto &after_befores : ops) {
    Op *after                        = after_befores.first;
    const std::vector<Op *> &befores = after_befores.second;
    for (auto &before : befores) {
      insert(before, after);
    }
  }
}

namespace {
template <typename T>
std::vector<T> getValsOrEmpty(T key, const std::map<T, std::set<T>> &M) {
  auto found = M.find(key);
  if (found == M.end()) {
    return {};
  }
  std::vector<T> vals;
  vals.reserve(found->second.size());
  for (T after : found->second) {
    vals.push_back(after);
  }
  return vals;
}
} // namespace

std::vector<Op *> TopoCons::getAfters(Op *before) const {
  return getValsOrEmpty(before, valsAfter);
};

std::vector<Op *> TopoCons::getBefores(Op *after) const {
  return getValsOrEmpty(after, valsBefore);
};

void TopoCons::transfer(Op *beforeTransfer, Op *afterTransfer) {

  if (!getBefores(beforeTransfer).empty() ||
      !getAfters(beforeTransfer).empty()) {
    logging::ir::debug("Transfering topological constraints from {} to {}",
                       beforeTransfer->str(),
                       afterTransfer->str());
  }

  // for all b : b -> beforeTransfer, insert
  //             b -> afterTransfer. The edge
  //             b -> beforeTransfer will be removed at the end
  for (Op *b : getBefores(beforeTransfer)) {
    insert(b, afterTransfer);
  }

  // for all a : beforeTransfer -> a, insert
  //             afterTransfer -> a. The edge
  //             beforeTransfer -> a will be removed at the end
  for (Op *a : getAfters(beforeTransfer)) {
    insert(afterTransfer, a);
  }

  remove(beforeTransfer);
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

// insert the topological constraint before -> after
void TopoCons::insert(Op *before, Op *after) {

  logging::ir::debug("Inserting topological constraints from {} to {}",
                     before->str(),
                     after->str());

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

  found = valsAfter.find(before);
  if (found != valsAfter.end()) {
    found->second.insert(after);
  } else {
    valsAfter[before] = {after};
  }

  found = valsBefore.find(after);
  if (found != valsBefore.end()) {
    found->second.insert(before);
  } else {
    valsBefore[after] = {before};
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

    os << fmt::format("    {}:\n", op->debugName());
    for (auto o : opsAfter) {
      os << fmt::format("      {}\n", o->debugName());
    }
  }

  os << "  valsBefore:\n";
  for (auto &op_opsBefore : tc.valsBefore) {
    auto op         = op_opsBefore.first;
    auto &opsBefore = op_opsBefore.second;

    os << fmt::format("    {}:\n", op->debugName());
    for (auto o : opsBefore) {
      os << fmt::format("      {}\n", o->debugName());
    }
  }

  return os;
}

} // namespace popart
