#include <poponnx/dataflow.hpp>
#include <poponnx/error.hpp>

namespace willow {

DataFlow::DataFlow()
    : batchesPerStep_(0), batchSize_(0), art_(AnchorReturnType::FINAL) {}

DataFlow::DataFlow(int BpR,
                   int bs,
                   const std::vector<TensorId> &v,
                   AnchorReturnType artIn_)
    : batchesPerStep_(BpR), batchSize_(bs), v_anchors(v), art_(artIn_) {
  for (auto &id : v_anchors) {
    s_anchors.insert(id);
  }
  if (art_ != AnchorReturnType::ALL) {
    throw error("Only ALL AnchorReturnType is currently supported");
  }
}

bool DataFlow::isAnchored(TensorId id) const {
  return (s_anchors.count(id) != 0);
}

} // namespace willow
