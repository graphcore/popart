// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <vector>
#include <poprithms/util/stringutil.hpp>
#include <popart/dataflow.hpp>
#include <popart/error.hpp>
#include <popart/sessionoptions.hpp>

namespace popart {

AnchorReturnType::AnchorReturnType(std::string artString,
                                   TileSet tileSet,
                                   ExchangeStrategy exchangeStrategy)
    : artStr_(artString), artId_(getIdFromStr(artString)), returnPeriod_(0),
      tileSet_(tileSet), exchangeStrategy_(exchangeStrategy) {
  if (id() == AnchorReturnTypeId::EveryN) {
    throw error("Must specify return period with option 'EVERYN'");
  }
}

AnchorReturnType::AnchorReturnType(std::string artString,
                                   int returnPeriod,
                                   TileSet tileSet,
                                   ExchangeStrategy exchangeStrategy)
    : artStr_(artString), artId_(getIdFromStr(artString)),
      returnPeriod_(returnPeriod), tileSet_(tileSet),
      exchangeStrategy_(exchangeStrategy) {
  if (id() == AnchorReturnTypeId::EveryN) {
    if (returnPeriod_ <= 0) {
      throw error("Anchor return period must be greater than zero");
    }
  } else {
    throw error("A return period should not be supplied for this anchor "
                "return type");
  }
}

int AnchorReturnType::rp() const {
  if (id() == AnchorReturnTypeId::EveryN)
    return returnPeriod_;
  else
    throw error("A return period should not be supplied for this anchor "
                "return type");
}

AnchorReturnTypeId AnchorReturnType::getIdFromStr(std::string artString) {
  auto tempStr = poprithms::util::lowercase(artString);
  if (tempStr == "final")
    return AnchorReturnTypeId::Final;
  else if (tempStr == "everyn")
    return AnchorReturnTypeId::EveryN;
  else if (tempStr == "all")
    return AnchorReturnTypeId::All;
  else if (tempStr == "sum")
    return AnchorReturnTypeId::Sum;
  else
    throw error("Invalid anchor return type ID supplied: " + artString);
}

std::ostream &operator<<(std::ostream &oss, AnchorReturnTypeId art) {
  switch (art) {
  case (AnchorReturnTypeId::Final): {
    oss << "Final";
    break;
  }

  case (AnchorReturnTypeId::Sum): {
    oss << "Sum";
    break;
  }

  case (AnchorReturnTypeId::EveryN): {
    oss << "EveryN";
    break;
  }

  case (AnchorReturnTypeId::All): {
    oss << "All";
    break;
  }
  }
  return oss;
}

std::size_t AnchorReturnType::hash() const {
  return std::hash<std::string>()(artStr_) ^ std::hash<int>()(returnPeriod_) ^
         std::hash<int>()(static_cast<int>(tileSet_)) ^
         std::hash<int>()(static_cast<int>(exchangeStrategy_));
}

InputSettings::InputSettings()
    : tileSet_(TileSet::Compute),
      exchangeStrategy_(ExchangeStrategy::JustInTime) {}

InputSettings::InputSettings(TileSet tileSet, ExchangeStrategy exchangeStrategy)
    : tileSet_(tileSet), exchangeStrategy_(exchangeStrategy) {}

DataFlow::DataFlow() : batchesPerStep_(0) {}

DataFlow::DataFlow(int bps) : batchesPerStep_(bps) {}

DataFlow::DataFlow(int bps, const AnchorReturnTypeMap &m)
    : batchesPerStep_(bps), m_anchors(m) {

  if (batchesPerStep_ <= 0) {
    throw error("'Batches per step' must be greater than zero");
  }

  for (auto &id_rt : m_anchors) {
    v_anchors.push_back(id_rt.first);
    s_anchors.insert(id_rt.first);
    isValidAnchorReturnPeriod(id_rt.first, batchesPerStep_);

    // Compile unique list of return periods for all anchors,
    // used when building the graph so that a minimum of Tensors
    // are added to track batch count.
    // Don't track batch count for 'ALL' or 'FINAL' return types
    if (id_rt.second.id() == AnchorReturnTypeId::EveryN) {
      int rp = id_rt.second.rp();
      if (std::find(v_rps.begin(), v_rps.end(), rp) == v_rps.end()) {
        v_rps.push_back(rp);
      }
    }
  }
}

static AnchorReturnTypeMap anchorMapFromVector(const std::vector<TensorId> tIds,
                                               const AnchorReturnType &art) {
  AnchorReturnTypeMap anchor_map;
  for (const auto &id : tIds) {
    anchor_map.emplace(id, art);
  }
  return anchor_map;
}

DataFlow::DataFlow(int bps,
                   const std::vector<TensorId> tIds,
                   const AnchorReturnType &art)
    : DataFlow(bps, anchorMapFromVector(tIds, art)) {}

bool DataFlow::isAnchored(TensorId id) const {
  return (s_anchors.count(id) != 0);
}

bool DataFlow::isBatchCountingRequired() const { return (!rps().empty()); }

AnchorReturnType DataFlow::art(TensorId anchorId) const {
  if (!isAnchored(anchorId)) {
    throw error("Tensor " + anchorId.str() + " is not an anchor");
  }

  return m_anchors.at(anchorId);
}

unsigned DataFlow::numOutFetchesPerRepl(const struct SessionOptions &opts,
                                        const TensorId &anchorId) const {
  if (!isAnchored(anchorId)) {
    return 0;
  } else {
    const auto rt = m_anchors.at(anchorId);
    const int accumFactor =
        opts.enableGradientAccumulation ? opts.accumulationFactor : 1;

    switch (rt.id()) {
    case AnchorReturnTypeId::Final:
    case AnchorReturnTypeId::Sum: {
      return 1;
    }
    case AnchorReturnTypeId::EveryN: {
      auto batches = batchesPerStep_ / rt.rp();
      return batches * accumFactor;
    }
    case AnchorReturnTypeId::All: {
      return batchesPerStep_ * accumFactor;
    }
    default: {
      throw error("Unsupported AnchorReturnTypeId ({})", rt.id());
    }
    }
  }
}

void DataFlow::isValidAnchorReturnPeriod(TensorId anchorId,
                                         int batchesPerStep) {
  if (art(anchorId).id() == AnchorReturnTypeId::EveryN) {
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

std::size_t DataFlow::hash() const {
  auto hash = std::hash<int>()(batchesPerStep());
  for (auto tid_art : m_anchors) {
    hash = hash ^ std::hash<TensorId>()(tid_art.first) ^
           std::hash<AnchorReturnType>()(tid_art.second);
  }
  return hash;
}

} // namespace popart
