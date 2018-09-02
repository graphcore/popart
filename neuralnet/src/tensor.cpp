#include <neuralnet/error.hpp>
#include <neuralnet/graph.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

int Consumers::n(const Op *op) const {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    return 0;
  } else {
    return found->second;
  }
}

const std::map<const Op *, int> &Consumers::getMap() const {
  return consumers_m;
}

void Consumers::extend(const std::map<const Op *, int> &m) {
  for (auto &op_count : m) {
    auto found = consumers_m.find(op_count.first);
    if (found != consumers_m.end()) {
      found->second += op_count.second;
    } else {
      consumers_m[op_count.first] = op_count.second;
    }
  }
}

Tensor::Tensor(TensorId n, TensorType t, Graph *g)
    : id(n), pgraph(g), type(t), tensor_type(pgraph->tensorTypes.asString(t)),
      producer(nullptr) {}

void Consumers::decrement(const Op *op) {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    throw error("cannot decrement non-existant consumer");
  }
  --(found->second);
  if (found->second == 0) {
    consumers_m.erase(op);
  }
}

void Consumers::increment(const Op *op) {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    consumers_m[op] = 1;
  } else {
    ++(found->second);
  }
}

} // namespace neuralnet
