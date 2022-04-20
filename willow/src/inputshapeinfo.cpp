// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <iosfwd>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <popart/error.hpp>
#include <popart/inputshapeinfo.hpp>

#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"

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

namespace std {
std::size_t std::hash<popart::InputShapeInfo>::operator()(
    const popart::InputShapeInfo &info) const {
  std::size_t seed = 0;
  for (const auto &kv : info.getInfos()) {
    boost::hash_combine(seed, kv.first);

    std::stringstream ss;
    kv.second.append(ss);
    std::string serializedTensorInfo = ss.str();
    boost::hash_combine(seed, serializedTensorInfo);
  }
  return seed;
}
} // namespace std
