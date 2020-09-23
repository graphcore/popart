// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <popart/op.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

namespace popart {

TensorId TensorIndexMap::id(int index) const { return tensor(index)->id; }

std::map<int, Shape> TensorIndexMap::getIndexShapeMap() {
  auto &M = tensorMap();
  std::map<int, Shape> outMap;
  for (auto index_tensor : M) {
    auto index  = index_tensor.first;
    auto tensor = index_tensor.second;
    if (!tensor->info.isSet()) {
      throw error("Tensor info not set in TensorIndexMap::getIndexShapeMap()");
    }
    outMap[index] = tensor->info.shape();
  }
  return outMap;
}

TensorIndexMap::~TensorIndexMap() = default;

std::vector<TensorId> TensorIndexMap::getSerialised() const {
  int maxIndex = 0;
  for (auto &ind_tensor : tensor_map) {
    if (ind_tensor.first > maxIndex) {
      maxIndex = ind_tensor.first;
    }
  }
  std::vector<TensorId> serialised(maxIndex + 1, "");
  for (auto &ind_tensor : tensor_map) {
    serialised[ind_tensor.first] = ind_tensor.second->id;
  }
  return serialised;
}

bool TensorIndexMap::hasIndex(int index) const {
  return tensor_map.find(index) != tensor_map.end();
}

void TensorIndexMap::setInfoIfIndex(const TensorInfo &info_, int index) {
  if (hasIndex(index)) {
    tensor(index)->info = info_;
  }
}

const std::map<int, Tensor *> &TensorIndexMap::tensorMap() const {
  return tensor_map;
}

std::map<int, TensorId> TensorIndexMap::tensorIdMap() const {
  std::map<int, TensorId> M;
  for (auto &index_tensor : tensorMap()) {
    M[index_tensor.first] = index_tensor.second->id;
  }
  return M;
}

int TensorIndexMap::n() const { return static_cast<int>(tensor_map.size()); }

void TensorIndexMap::append(std::stringstream &ss,
                            std::string prefix,
                            int max_id_length) const {
  int index = 0;

  for (auto &index_tensor : tensor_map) {
    ss << prefix << '@' << index_tensor.first << ':' << ' '
       << padded(index_tensor.second->id, max_id_length + 1)

       << ' ' << padded(index_tensor.second->tensor_type(), 9);
    if (index_tensor.second->info.isSet()) {
      ss << ' ';
      index_tensor.second->info.append(ss);
    }

    ++index;
    if (index != tensor_map.size()) {
      ss << '\n';
    }
  }
}

int TensorIndexMap::maxIdLength() const {
  int max_id_length = 0;
  for (const auto &tensor_indices : indicesMap()) {
    max_id_length = std::max(max_id_length,
                             static_cast<int>(tensor_indices.first->id.size()));
  }
  return max_id_length;
}

void TensorIndexMap::insert(int index, Tensor *ptensor) {
  tensor_map[index] = ptensor;
  auto found        = indices_map.find(ptensor);
  if (found == indices_map.end()) {
    indices_map[ptensor] = {index};
  } else {
    indices_map[ptensor].push_back(index);
  }
}

void TensorIndexMap::reset(int index, Tensor *ptensor) {
  auto previous = tensor_map[index];

  tensor_map[index] = ptensor;

  if (indices_map.find(ptensor) == indices_map.end()) {
    indices_map[ptensor] = {};
  }
  indices_map[ptensor].push_back(index);

  // clean up previous tensor
  std::vector<int> newIndices;
  for (auto &ind : indices_map[previous]) {
    if (ind != index) {
      newIndices.push_back(ind);
    }
  }
  if (newIndices.size() != 0) {
    indices_map[previous] = newIndices;
  } else {
    indices_map.erase(previous);
  }
}

bool TensorIndexMap::contains(Tensor *t) const {
  auto it = indices_map.find(t);
  return it != indices_map.end();
}

void TensorIndexMap::erase(int index) {
  const auto tm_itr = tensor_map.find(index);

  if (tm_itr != tensor_map.end()) {
    const auto im_itr = indices_map.find(tm_itr->second);

    auto &imv    = im_itr->second;
    auto imv_itr = std::find(imv.begin(), imv.end(), index);

    // Remove the index from indices_map.
    if (imv_itr != imv.end()) {
      imv.erase(imv_itr);
    }

    // If the Tensor has no indices, remove it from indices_map.
    if (imv.empty()) {
      indices_map.erase(im_itr);
    }

    // Remove the tensor from the tensor_map.
    tensor_map.erase(tm_itr);
  }
}

void TensorIndexMap::clear() {
  tensor_map.clear();
  indices_map.clear();
}

Tensor *TensorIndexMap::tensor(int index) { return tensor_map.at(index); }

const Tensor *TensorIndexMap::tensor(int index) const {
  return tensor_map.at(index);
}

const std::vector<int> &TensorIndexMap::indices(Tensor *ptensor) const {
  return indices_map.at(ptensor);
}

const std::vector<Tensor *> TensorIndexMap::tensors() const {
  std::vector<Tensor *> tensors;
  for (const auto &tensor_indices : indicesMap()) {
    tensors.push_back(tensor_indices.first);
  }

  // ensure that the order of the returned vector does not depend on the
  // addresses of the Tensors in memory
  std::sort(tensors.begin(),
            tensors.end(),
            [](const Tensor *a, const Tensor *b) { return a->id < b->id; });

  return tensors;
}

} // namespace popart
