#include <willow/error.hpp>
#include <willow/ir.hpp>
#include <willow/onnxutil.hpp>
#include <willow/tensor.hpp>
#include <willow/util.hpp>

namespace willow {

Consumers::Consumers(Tensor *tensorConsumed_)
    : tensorConsumed(tensorConsumed_) {
  // currently this class variable is not used,
  // to prevent a warning about unused class variable
  // we do this
  (void)tensorConsumed;
}

int Consumers::n(Op *op) const {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    return 0;
  } else {
    return found->second;
  }
}

std::vector<Op *> Consumers::consumersWhichTopoBefore(Op *op) const {
  auto found0 = consumers_m.find(op);
  if (found0 == consumers_m.end()) {
    throw error("Op " + std::to_string(op->id) + " is not a consumer");
  }

  // if it is constrained to be the last consumer, then all
  // other consumers must come before it
  else if (op == topoLast) {
    std::vector<Op *> before;
    before.reserve(consumers_m.size() - 1);
    for (auto &consumer : getOps()) {
      if (consumer != op) {
        before.push_back(consumer);
      }
    }
    return before;
  }

  // Note : we need more fine grained topo control.
  // The advantage of Last is that we don't
  // need to worry about new consumers being added,
  // very useful for the VarUpdate ops. Previously
  // we had a First too, but decided this was not
  // useful (inplace needs more precision than just
  // First when inplace ops appear sequentially)
  else {
    return {};
  }
}

TensorData *Tensor::tensorData() {
  if (data_.get() == nullptr) {
    throw error("Data not set for " + id);
  }
  return data_.get();
}

TensorData::TensorData(const onnx::TensorProto &tp) {
  TensorInfo info(tp);
  data_.resize(info.nbytes());
  std::memcpy(data_.data(), onnxutil::getData(tp), info.nbytes());
}

TensorData::TensorData(const TensorInfo &info, const void *from) {
  data_.resize(info.nbytes());
  std::memcpy(data_.data(), from, info.nbytes());
}

void *TensorData::data() { return data_.data(); }

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

bool Consumers::hasTopoLast() const { return topoLast != nullptr; }

Op *Consumers::getTopoLast() const {
  if (!hasTopoLast()) {
    throw error("no topologically last op for consumers");
  }
  return topoLast;
}

bool Consumers::hasWeakTopoCons() const {
  // no weak topo cons implemented yet
  return false;
}

void Consumers::setTopoLast(Op *op) {
  if (topoLast != nullptr) {
    throw error("cannot set topo last when one already exists");
  }
  topoLast = op;
}

void Consumers::removeTopoLast() { topoLast = nullptr; }

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
                " as not one deja");
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
Tensor::Tensor(TensorId n, TensorType t, Ir *g)
    : Vertex(), id(n), pir(g), consumers(this), producer(nullptr),
      tensorTypeInfo(&getTensorTypeInfoMap().at(t)) {}

void Consumers::decrement(Op *op) {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    throw error("cannot decrement non-existant consumer, " + op->str());
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

} // namespace willow
