#include <neuralnet/error.hpp>
#include <neuralnet/graph.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

int Consumers::n(Op *op) const {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    return 0;
  } else {
    return found->second;
  }
}

const std::map<Op *, int> &Consumers::getMap() const { return consumers_m; }

void Consumers::extend(const std::map<Op *, int> &m) {
  for (auto &op_count : m) {
    auto found = consumers_m.find(op_count.first);
    if (found != consumers_m.end()) {
      found->second += op_count.second;
    } else {
      consumers_m[op_count.first] = op_count.second;
    }
  }
}

int Consumers::getTotal() const {
  //  using X = decltype(consumers_m.begin());
  //  return std::accumulate(consumers_m.begin(), consumers_m.end(), 0,
  //      [](const X & v1, const X & v2){return v1.second + v2.second;});
  int total = 0;
  for (auto &op_count : consumers_m) {
    total += op_count.second;
  }
  return total;
}

Tensor::Tensor(TensorId n, TensorType t, Graph *g)
    : id(n), pgraph(g), producer(nullptr),
      tensorTypeInfo(&getTensorTypeInfoMap().at(t)) {}

void Consumers::decrement(Op *op) {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    throw error("cannot decrement non-existant consumer");
  }
  --(found->second);
  if (found->second == 0) {
    consumers_m.erase(op);
  }
}

Op *Tensor::getProducer() {
  if (!hasProducer()) {
    throw error("No producer for tensor " + id + " to return");
  }
  return producer;
}

bool Tensor::hasProducer() const { return producer != nullptr; }

void Consumers::increment(Op *op) {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    consumers_m[op] = 1;
  } else {
    ++(found->second);
  }
}

std::vector<Op *> Consumers::getOps() {
  std::vector<Op *> ops;
  ops.reserve(consumers_m.size());
  for (auto &x : consumers_m) {
    ops.push_back(x.first);
  }
  return ops;
}

const std::map<Tensor *, std::vector<int>> &TensorIndexMap::indicesMap() const {
  return indices_map;
}

const std::map<TensorType, TensorTypeInfo> &getTensorTypeInfoMap() {
  static std::map<TensorType, TensorTypeInfo> M = initTensorTypeInfoMap();
  return M;
}

TensorType Tensor::tensorType() const { return tensorTypeInfo->type(); }

const std::string &Tensor::tensor_type() const {
  return tensorTypeInfo->type_s();
}

TensorType TensorTypeInfo::type() const { return tensorType_; }

const std::string &TensorTypeInfo::type_s() const { return tensor_type_; }

TensorTypeInfo::TensorTypeInfo(TensorType t_, std::string ts_)
    : tensorType_(t_), tensor_type_(ts_) {}

std::map<TensorType, TensorTypeInfo> initTensorTypeInfoMap() {
  std::map<TensorType, TensorTypeInfo> tensor_types_m = {
      {TensorType::ActGrad, {TensorType::ActGrad, "ActGrad"}},
      {TensorType::Const, {TensorType::Const, "Const"}},
      //{TensorType::Gradient, {TensorType::Gradient, "Gradient"}},
      {TensorType::Momentum, {TensorType::Momentum, "Momentum"}},
      //{TensorType::Other, {TensorType::Other, "Other"}},
      {TensorType::Stream, {TensorType::Stream, "Stream"}},
      {TensorType::Unknown, {TensorType::Unknown, "Unknown"}},
      {TensorType::Variable, {TensorType::Variable, "Variable"}}};
  if (tensor_types_m.size() != static_cast<int64_t>(TensorType::N)) {
    throw error("missing element in TensorTypes");
  }
  return tensor_types_m;
}

} // namespace neuralnet
