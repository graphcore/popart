#include <algorithm>
#include <cstring>

#include <poponnx/error.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/op.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensorindex.hpp>
#include <poponnx/util.hpp>

namespace poponnx {

std::unique_ptr<Tensor> Tensor::clone() const {
  std::unique_ptr<Tensor> theClone(new Tensor("clone_" + id, tensorType(), ir));
  theClone->info = info;
  return theClone;
}

Consumers::Consumers(Tensor *tensorConsumed_)
    : tensorConsumed(tensorConsumed_) {}

int Consumers::n(Op *op) const {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    return 0;
  } else {
    return found->second;
  }
}

bool Tensor::hasTensorData() const {
  if (data_.get() == nullptr) {
    return false;
  }
  return true;
}

TensorData *Tensor::tensorData() {
  if (data_.get() == nullptr) {
    throw error("Data not set for " + id);
  }
  return data_.get();
}

void Consumers::append(std::stringstream &ss) {
  std::string tab = "     ";

  ss << '\n';
  ss << "Consumer count of Tensor " << tensorConsumed->id << " : " << '\n';
  int max_length = 0;
  for (auto &op_count : getMap()) {
    max_length =
        std::max(max_length, static_cast<int>(op_count.first->str().size()));
  }

  for (auto &op_count : getMap()) {
    ss << padded(op_count.first->str(), max_length + 1) << " : "
       << op_count.second << '\n';
  }
  ss << "Total number of consumptions: " << getTotal();
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

void Tensor::setProducer(Op *op) {
  if (hasProducer()) {
    throw error("Cannot set a producer for Tensor " + id + " as already one");
  }
  producer = op;
}

void Tensor::resetProducer(Op *op) {
  if (!hasProducer()) {
    throw error("Cannot reset a producer for Tensor " + id +
                " as it does not already have one");
  }
  producer = op;
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

// using 'this' in a constructor list? Be careful.
// https://stackoverflow.com/questions/5058349
Tensor::Tensor(TensorId n, TensorType t, Ir &g)
    : Vertex(), id(n), consumers(this), ir(g), producer(nullptr),
      tensorTypeInfo(&getTensorTypeInfoMap().at(t)), data_(nullptr) {
  // ir is currently unused - this removes the compiler warning
  (void)ir;
}

void Consumers::decrement(Op *op) {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    throw error("cannot decrement non-existant consumer, " + op->debugName());
  }
  --(found->second);
  if (found->second == 0) {
    consumers_m.erase(op);
  }
}

Op *Tensor::getProducer() const {
  if (!hasProducer()) {
    throw error("No producer for tensor " + id + " to return");
  }
  return getProducerUnsafe();
}

Op *Tensor::getProducerUnsafe() const { return producer; }

bool Tensor::hasProducer() const { return producer != nullptr; }

void Consumers::increment(Op *op) {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    consumers_m[op] = 1;
  } else {
    ++(found->second);
  }
}

std::vector<Op *> Consumers::getOps() const {
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

void Tensor::setTensorType(TensorType t) {
  tensorTypeInfo = &getTensorTypeInfoMap().at(t);
}

std::vector<Op *> Tensor::associatedOps() const {
  std::vector<Op *> result = consumers.getOps();

  if (hasProducer()) {
    result.push_back(getProducer());
  }

  return result;
}

TensorType TensorTypeInfo::type() const { return tensorType_; }

const std::string &TensorTypeInfo::type_s() const { return tensor_type_; }

TensorTypeInfo::TensorTypeInfo(TensorType t_, std::string ts_)
    : tensorType_(t_), tensor_type_(ts_) {}

std::map<TensorType, TensorTypeInfo> initTensorTypeInfoMap() {
  std::map<TensorType, TensorTypeInfo> tensor_types_m = {
      {TensorType::ActGrad, {TensorType::ActGrad, "ActGrad"}},
      {TensorType::Const, {TensorType::Const, "Const"}},
      {TensorType::Momentum, {TensorType::Momentum, "Momentum"}},
      {TensorType::Stream, {TensorType::Stream, "Stream"}},
      {TensorType::Unknown, {TensorType::Unknown, "Unknown"}},
      {TensorType::Variable, {TensorType::Variable, "Variable"}}};
  if (tensor_types_m.size() != static_cast<int64_t>(TensorType::N)) {
    throw error("missing element in TensorTypes");
  }
  return tensor_types_m;
}

VariableTensor::VariableTensor(TensorId n, Ir &g)
    : Tensor(n, TensorType::Variable, g),
      variableUpdateType(VariableUpdateType::Gradient) {}

std::unique_ptr<Tensor> VariableTensor::clone() const {
  std::unique_ptr<Tensor> theClone(new VariableTensor("clone_" + id, ir));
  theClone->info = info;
  return theClone;
}

} // namespace poponnx
