// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <set>
#include <popart/error.hpp>
#include <popart/inputshapeinfo.hpp>

namespace popart {

void InputShapeInfo::add(TensorId id, const TensorInfo &info) {
  infos[id] = info;
}

const TensorInfo &InputShapeInfo::get(TensorId id) const {
  auto found = infos.find(id);
  if (found == infos.end()) {
    throw error("No input shape info for Tensor `{}' in InputShapeInfo::get",
                id);
  }
  return found->second;
}

bool InputShapeInfo::has(TensorId id) const {
  return infos.find(id) != infos.end();
}

std::vector<TensorId> InputShapeInfo::getAllTensorIds() const {
  // we first put the TensorIds into a set, so that duplication is removed
  std::set<TensorId> all_;
  for (const auto &id_info : infos) {
    all_.insert(id_info.first);
  }

  // then any other TensorIds from other maps in InputShapeInfo will be added
  // to the set here.

  std::vector<TensorId> all;
  for (auto &x : all_) {
    all.push_back(x);
  }
  return all;
}

} // namespace popart
