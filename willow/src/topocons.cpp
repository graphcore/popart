#include <poponnx/op.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/topocons.hpp>

namespace poponnx {

void TopoCons::setFinalConsumer(const Tensor *tensor, Op *last) {

  if (tensor->consumers.n(last) == 0) {
    throw error("Cannot set " + last->str() + " as last consumer of " +
                tensor->id + ", as it is not a consumer.");
  }

  for (Op *op : tensor->consumers.getOps()) {
    if (op != last) {
      // insert the topological constraint, op -> last
      insert(op, last);
    }
  }

  for (Op *op : tensor->consumers.getOps()) {
    if (op != last) {
      if (contains(last, op)) {
        throw error("Failure setting " + last->str() + " to last: " +
                    op->str() + " is constrained to be before. " +
                    "setTopoLast does not " + "remove existing constraints.");
      }
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
  auto found = valsAfter.find(before);
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

} // namespace poponnx
