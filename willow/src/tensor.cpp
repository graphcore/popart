#include <cstring>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/util.hpp>

namespace willow {

void Consumers::takeFrom(Consumers &giver) {
  // we first confirm that no consumers are shared,
  // Using O(N^2) algorithm, could be O(N.log(N)).
  for (Op *op_taker : getOps()) {
    for (Op *op_giver : giver.getOps()) {
      if (op_taker == op_giver) {
        throw error("Cannot transfer consumers from " +
                    giver.tensorConsumed->id + " to " + tensorConsumed->id +
                    " as they have " + op_taker->str() + " in common");
      }
    }
  }

  for (Op *op : giver.getOps()) {
    consumers_m[op] = giver.consumers_m[op];
    topoCons[op]    = giver.topoCons[op];

    // set the input of op to be this->tensorConsumed
    for (auto index_tensor : op->input.tensorMap()) {
      InIndex inIndex = index_tensor.first;
      Tensor *tensor  = index_tensor.second;
      if (tensor == giver.tensorConsumed) {
        op->input.reset(inIndex, tensorConsumed);
      }
    }
  }
}

void Consumers::takeTopoCons(Op *beforeTransfer, Op *afterTransfer) {

  // collect all the map keys
  std::vector<Op *> keys;
  for (auto key_vals : topoCons) {
    keys.push_back(key_vals.first);
  }

  // vals: replace all occurrences of beforeTransfer with afterTrasfer
  for (Op *key : keys) {
    for (Op *&val : topoCons[key]) {
      if (val == beforeTransfer) {
        val = afterTransfer;
      }
    }
  }

  // keys: replace all occurrences of beforeTransfer with afterTrasfer
  for (Op *key : keys) {
    if (key == beforeTransfer) {
      topoCons[afterTransfer] = topoCons[beforeTransfer];
      topoCons.erase(beforeTransfer);
    }
  }
}

void Consumers::removeTopoCons(Op *op) {
  // if op is a key erase.
  if (topoCons.find(op) == topoCons.end()) {
    topoCons.erase(op);
  }

  // for all keys, remove op from the values vector, if present.
  // if after removing op the vector is empty. erase the key.
  for (Op *op2 : getOps()) {
    std::vector<Op *> newCons;
    for (Op *op3 : topoCons[op2]) {
      if (op3 != op) {
        newCons.push_back(op3);
      }
    }
    if (newCons.size() == 0) {
      topoCons.erase(op2);
    } else {
      topoCons[op2] = newCons;
    }
  }
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

void Consumers::insertTopoCon(Op *before, Op *after) {
  // topoCons is of type "OpsBeforeKey"
  if (topoCons.find(after) == topoCons.end()) {
    topoCons[after] = {};
  }

  // if the topo con is already present, bail
  if (std::find(topoCons[after].begin(), topoCons[after].end(), before) !=
      topoCons[after].end()) {
    throw error("Already registered constraint [" + before->str() + " before " +
                after->str() + "]");
  }
  topoCons[after].push_back(before);
}

void Consumers::setTopoLast(Op *last) {
  if (n(last) == 0) {
    throw error("Cannot set " + last->str() + " as last consumer of " +
                tensorConsumed->id + " as it not a consumer.");
  }
  topoCons[last] = {};
  for (Op *op : getOps()) {
    if (op != last) {
      topoCons[last].push_back(op);
    }
  }

  for (Op *op : getOps()) {
    if (op != last) {
      if (std::find(topoCons[op].begin(), topoCons[op].end(), last) !=
          topoCons[op].end()) {
        throw error("Failure setting " + last->str() +
                    " to last: " + op->str() +
                    " is constrained to be before. setTopoLast does not " +
                    "remove existing constraints.");
      }
    }
  }
}

std::vector<Op *> Consumers::consumersWhichTopoBefore(Op *op) const {
  auto found0 = consumers_m.find(op);
  if (found0 == consumers_m.end()) {
    throw error("Op " + op->str() + " is not a consumer of " +
                tensorConsumed->id);
  }

  auto found = topoCons.find(op);
  if (found == topoCons.end()) {
    return {};
  } else {
    return found->second;
  }
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

bool Consumers::hasTopoCons() const { return topoCons.size() != 0; }

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
      tensorTypeInfo(&getTensorTypeInfoMap().at(t)) {
  // ir is currently unused - this removes the compiler warning
  (void)ir;
}

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

Op *Tensor::getProducer() const {
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

} // namespace willow
