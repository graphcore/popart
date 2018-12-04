#include <algorithm>
#include <vector>
#include <poponnx/dataflow.hpp>
#include <poponnx/error.hpp>

namespace poponnx {

AnchorReturnType::AnchorReturnType(std::string artString)
    : artId_(getIdFromStr(artString)), returnFrequency_(0) {
  if (id() == AnchorReturnTypeId::EVERYN) {
    throw error("Must specify return frequency with option 'EVERYN'");
  }
}

AnchorReturnType::AnchorReturnType(std::string artString, int returnFrequency)
    : artId_(getIdFromStr(artString)), returnFrequency_(returnFrequency) {
  if (id() == AnchorReturnTypeId::EVERYN) {
    if (returnFrequency_ <= 0) {
      throw error("Anchor return frequency must be greater than zero");
    }
  } else {
    throw error("A return frequency should not be supplied for this anchor "
                "return type");
  }
}

int AnchorReturnType::rf() const {
  if (id() == AnchorReturnTypeId::EVERYN)
    return returnFrequency_;
  else
    throw error("A return frequency should not be supplied for this anchor "
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

DataFlow::DataFlow() : batchesPerStep_(0), batchSize_(0) {}

DataFlow::DataFlow(int BpR,
                   int bs,
                   const std::map<TensorId, AnchorReturnType> &m)
    : batchesPerStep_(BpR), batchSize_(bs), m_anchors(m) {
  for (auto &id_rt : m_anchors) {
    v_anchors.push_back(id_rt.first);
    s_anchors.insert(id_rt.first);
    isValidAnchorReturnFrequency(id_rt.first, batchesPerStep_);

    // Compile unique list of return frequencies for all anchors,
    // used when building the graph so that a minimum of Tensors
    // are added to track batch count
    int rf;
    switch (id_rt.second.id()) {
    case AnchorReturnTypeId::FINAL: {
      rf = batchesPerStep();
      break;
    }
    case AnchorReturnTypeId::EVERYN: {
      rf = id_rt.second.rf();
      break;
    }
    case AnchorReturnTypeId::ALL: {
      // Don't track batch count for this return type
      rf = 0;
      break;
    }
    }
    if (rf) {
      if (std::find(v_rfs.begin(), v_rfs.end(), rf) == v_rfs.end()) {
        v_rfs.push_back(rf);
      }
    }
  }
}

bool DataFlow::isAnchored(TensorId id) const {
  return (s_anchors.count(id) != 0);
}

bool DataFlow::isBatchCountingRequired() const { return (!rfs().empty()); }

AnchorReturnType DataFlow::art(TensorId anchorId) const {
  if (!isAnchored(anchorId)) {
    throw error("Tensor " + anchorId + " is not an anchor");
  }

  return m_anchors.at(anchorId);
}

void DataFlow::isValidAnchorReturnFrequency(TensorId anchorId,
                                            int batchesPerStep) {
  if (art(anchorId).id() == AnchorReturnTypeId::EVERYN) {
    if (art(anchorId).rf() > batchesPerStep) {
      throw error(
          "Return frequency must be <= to the number of batches per step");
    } else if (batchesPerStep % art(anchorId).rf() != 0) {
      // Design decision, otherwise the position of batch N
      // will vary between epochs when looping over multiple
      // epochs
      throw error("Return frequency must be a factor of the number of batches "
                  "per step");
    }
  }
}

} // namespace poponnx
