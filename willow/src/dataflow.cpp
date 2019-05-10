#include <algorithm>
#include <vector>
#include <poponnx/dataflow.hpp>
#include <poponnx/error.hpp>

namespace poponnx {

AnchorReturnType::AnchorReturnType(std::string artString)
    : artId_(getIdFromStr(artString)), returnPeriod_(0) {
  if (id() == AnchorReturnTypeId::EVERYN) {
    throw error("Must specify return period with option 'EVERYN'");
  }
}

AnchorReturnType::AnchorReturnType(std::string artString, int returnPeriod)
    : artId_(getIdFromStr(artString)), returnPeriod_(returnPeriod) {
  if (id() == AnchorReturnTypeId::EVERYN) {
    if (returnPeriod_ <= 0) {
      throw error("Anchor return period must be greater than zero");
    }
  } else {
    throw error("A return period should not be supplied for this anchor "
                "return type");
  }
}

int AnchorReturnType::rp() const {
  if (id() == AnchorReturnTypeId::EVERYN)
    return returnPeriod_;
  else
    throw error("A return period should not be supplied for this anchor "
                "return type");
}

AnchorReturnTypeId AnchorReturnType::getIdFromStr(std::string artString) {
  if (artString == "FINAL")
    return AnchorReturnTypeId::FINAL;
  else if (artString == "EVERYN")
    return AnchorReturnTypeId::EVERYN;
  else if (artString == "ALL")
    return AnchorReturnTypeId::ALL;
  else
    throw error("Invalid anchor return type ID supplied: " + artString);
}

DataFlow::DataFlow() : batchesPerStep_(0) {}

DataFlow::DataFlow(int BpR, const std::map<TensorId, AnchorReturnType> &m)
    : batchesPerStep_(BpR), m_anchors(m) {
  for (auto &id_rt : m_anchors) {
    v_anchors.push_back(id_rt.first);
    s_anchors.insert(id_rt.first);
    isValidAnchorReturnPeriod(id_rt.first, batchesPerStep_);

    // Compile unique list of return periods for all anchors,
    // used when building the graph so that a minimum of Tensors
    // are added to track batch count.
    // Don't track batch count for 'ALL' or 'FINAL' return types
    if (id_rt.second.id() == AnchorReturnTypeId::EVERYN) {
      int rp = id_rt.second.rp();
      if (std::find(v_rps.begin(), v_rps.end(), rp) == v_rps.end()) {
        v_rps.push_back(rp);
      }
    }
  }
}

bool DataFlow::isAnchored(TensorId id) const {
  return (s_anchors.count(id) != 0);
}

bool DataFlow::isBatchCountingRequired() const { return (!rps().empty()); }

AnchorReturnType DataFlow::art(TensorId anchorId) const {
  if (!isAnchored(anchorId)) {
    throw error("Tensor " + anchorId + " is not an anchor");
  }

  return m_anchors.at(anchorId);
}

void DataFlow::isValidAnchorReturnPeriod(TensorId anchorId,
                                         int batchesPerStep) {
  if (art(anchorId).id() == AnchorReturnTypeId::EVERYN) {
    if (art(anchorId).rp() > batchesPerStep) {
      throw error("Return period must be <= to the number of batches per step");
    } else if (batchesPerStep % art(anchorId).rp() != 0) {
      // Design decision, otherwise the position of batch N
      // will vary between epochs when looping over multiple
      // epochs
      throw error("Return period must be a factor of the number of batches "
                  "per step");
    }
  }
}

} // namespace poponnx
