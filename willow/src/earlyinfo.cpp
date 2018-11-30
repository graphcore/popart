#include <set>
#include <poponnx/earlyinfo.hpp>
#include <poponnx/error.hpp>

namespace poponnx {

void EarlyInfo::add(TensorId id, const TensorInfo &info) { infos[id] = info; }

const TensorInfo &EarlyInfo::get(TensorId id) const {
  auto found = infos.find(id);
  if (found == infos.end()) {
    throw error("No early info for " + id);
  }
  return found->second;
}

bool EarlyInfo::has(TensorId id) const { return infos.find(id) != infos.end(); }

std::vector<TensorId> EarlyInfo::getAllTensorIds() const {
  // we first put the TensorIds into a set, so that duplication is removed
  std::set<TensorId> all_;
  for (const auto &id_info : infos) {
    all_.insert(id_info.first);
  }

  // then any other TensorIds from other maps in EarlyInfo will be added
  // to the set here.

  std::vector<TensorId> all;
  for (auto &x : all_) {
    all.push_back(x);
  }
  return all;
}

} // namespace poponnx
