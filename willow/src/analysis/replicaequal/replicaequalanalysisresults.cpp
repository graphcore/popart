// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <analysis/replicaequal/replicaequalanalysisresults.hpp>

namespace popart {

const ReplicaEqualAnalysisResults::Time ReplicaEqualAnalysisResults::initTime =
    -1;

ReplicaEqualAnalysisResults::ReplicaEqualAnalysisResults()
    : haveChange{false}, haveDisagreement{false}, results{}, opTimeMap{},
      graphSchedules{} {}

ReplicaEqualAnalysisResults::~ReplicaEqualAnalysisResults() = default;

bool ReplicaEqualAnalysisResults::containsAt(
    const Tensor *tensor,
    nonstd::optional<const Op *> op) const {
  assert(tensor != nullptr);

  // Time to use.
  Time time = (op) ? opTimeMap.at(*op) : initTime;

  // Check if tensor has values already.
  auto tensorIt = results.find(tensor);
  if (tensorIt == results.end()) {
    // Not any mapping for tensor.
    return false;
  } else {
    // There is a mapping already.
    auto &existingMap = tensorIt->second;

    // Find entry at exact time.
    auto timeIt = existingMap.find(time);
    return (timeIt != existingMap.end());
  }
}

bool ReplicaEqualAnalysisResults::containsBefore(const Tensor *tensor,
                                                 const Op *op) const {
  assert(tensor != nullptr);

  // Time to use.
  Time time = opTimeMap.at(op);

  // Check if tensor has values already.
  auto tensorIt = results.find(tensor);
  if (tensorIt == results.end()) {
    // Not any mapping for tensor.
    return false;
  } else {
    // There is a mapping already.
    auto &existingMap = tensorIt->second;

    // Find first entry with time <(opTime) using the fact that `existingMap` is
    // ordered in descending.
    auto timeIt = existingMap.upper_bound(time);
    return (timeIt != existingMap.end());
  }
}

bool ReplicaEqualAnalysisResults::setValueAt(const Tensor *tensor,
                                             nonstd::optional<const Op *> op,
                                             const IsReplicaEqual &value) {
  assert(tensor != nullptr);

  bool changed = true;

  // Local name to distinguish from existing values.
  const auto &newValue = value;

  // Time to use.
  Time time = (op) ? opTimeMap.at(*op) : initTime;

  // Check if tensor has values already.
  auto tensorIt = results.find(tensor);
  if (tensorIt == results.end()) {
    // No mapping for this tensor yet.
    results.insert(tensorIt, {tensor, {{time, value}}});
  } else {
    // There is a mapping already.
    auto &existingMap = tensorIt->second;
    auto timeIt       = existingMap.find(time);
    if (timeIt == existingMap.end()) {
      // No mapping for this time.
      existingMap.insert(timeIt, {time, value});
    } else {
      // Already have a value.
      IsReplicaEqual oldValue = timeIt->second;

      if (oldValue != newValue) {
        // Value differs -> take conjunction.
        timeIt->second = oldValue && newValue;
        if (timeIt->second == oldValue) {
          // We didn't change the value from false to true.
          haveDisagreement = true;
          disagreements.insert(tensor);
        } else {
          // We changed the value.
          haveChange = true;
        }
      } else {
        // We didn't change the result object.
        changed = false;
      }
    }
  }

  return changed;
}

IsReplicaEqual
ReplicaEqualAnalysisResults::getValueAt(const Tensor *tensor,
                                        nonstd::optional<const Op *> op) const {

  assert(tensor != nullptr);

  // Time to use.
  Time time = (op) ? opTimeMap.at(*op) : initTime;

  // Check if tensor has values already.
  auto tensorIt = results.find(tensor);
  if (tensorIt == results.end()) {
    throw internal_error("[ReplicaEqualAnalysisResults] No results for '{}'. "
                         "Result map is:\n\n{}",
                         tensor->id,
                         *this);
  } else {
    // There is a mapping already.
    auto &existingMap = tensorIt->second;

    // Find first entry with time <=(opTime-1) if op is set, or <=initTime if op
    // is not set, using the fact that `existingMap` is ordered in descending.
    auto timeIt = existingMap.find(time);
    if (timeIt == existingMap.end()) {
      // No mapping found.
      if (op) {
        throw internal_error("[ReplicaEqualAnalysisResults] No result for '{}' "
                             "(at {}). Result map is:\n\n{}",
                             tensor->id,
                             (*op)->debugName(),
                             *this);
      } else {
        throw internal_error("[ReplicaEqualAnalysisResults] No result for '{}' "
                             "(at initialisation). Result map is:\n\n{}",
                             tensor->id,
                             *this);
      }
    } else {
      return timeIt->second;
    }
  }
}

IsReplicaEqual ReplicaEqualAnalysisResults::getValueBefore(const Tensor *tensor,
                                                           const Op *op) const {

  assert(tensor != nullptr);

  // Time to use.
  Time time = opTimeMap.at(op);

  // Check if tensor has values already.
  auto tensorIt = results.find(tensor);
  if (tensorIt == results.end()) {
    throw internal_error("[ReplicaEqualAnalysisResults] No results for '{}'. "
                         "Result map is:\n{}",
                         tensor->id,
                         *this);
  } else {
    // There is a mapping already.
    auto &existingMap = tensorIt->second;

    // Find first entry with time <(opTime) using the fact that `existingMap` is
    // ordered in descending.
    auto timeIt = existingMap.upper_bound(time);
    if (timeIt == existingMap.end()) {
      // No mapping found.
      throw internal_error("[ReplicaEqualAnalysisResults] No result for '{}' "
                           "(before {}). Result map is:\n\n{}",
                           tensor->id,
                           op->debugName(),
                           *this);
    } else {
      return timeIt->second;
    }
  }
}

IsReplicaEqual
ReplicaEqualAnalysisResults::getFinalValue(const Tensor *tensor) const {

  assert(tensor != nullptr);

  // Check if tensor has values already.
  auto tensorIt = results.find(tensor);
  if (tensorIt == results.end()) {
    throw internal_error("[ReplicaEqualAnalysisResults] No results for '{}'. "
                         "Result map is:\n\n{}",
                         tensor->id,
                         *this);
  } else {
    // There is a mapping already.
    auto &existingMap = tensorIt->second;

    // Find value with the latest time, using the fact `existingMap` is ordered
    // in descending.
    auto timeIt = existingMap.begin();
    if (timeIt == existingMap.end()) {
      // No mapping found.
      throw internal_error("[ReplicaEqualAnalysisResults] No result for '{}'. "
                           "Result map is:\n\n{}",
                           tensor->id,
                           *this);
    } else {
      return timeIt->second;
    }
  }
}

bool ReplicaEqualAnalysisResults::hasChanged() const { return haveChange; }

bool ReplicaEqualAnalysisResults::hasDisagreements() const {
  return haveDisagreement;
}

const ReplicaEqualAnalysisResults::Disagreements &
ReplicaEqualAnalysisResults::getDisagreements() const {
  return disagreements;
}

void ReplicaEqualAnalysisResults::clearChanges() {
  haveChange       = false;
  haveDisagreement = false;
  disagreements.clear();
}

void ReplicaEqualAnalysisResults::setGraphSchedules(
    const GraphSchedules &graphSchedules_) {

  graphSchedules = graphSchedules_;
  // Set `opTimeMap` lookup that caches schedule locations for Ops.
  for (const auto &schedEntry : graphSchedules) {
    int schedLocation = 0;
    for (auto op : schedEntry.second) {
      opTimeMap[op] = schedLocation++;
    }
  }
}

std::ostream &operator<<(std::ostream &out,
                         const ReplicaEqualAnalysisResults &results) {
  // A dump of ReplicaEqualAnalysisResults to a string for debugging purposes.
  for (const auto &entry1 : results.results) {
    auto tensor   = entry1.first;
    auto &timeMap = entry1.second;
    out << "'" << tensor->id << "': [";
    bool isFirst = true;
    for (const auto &entry2 : timeMap) {
      if (!isFirst) {
        out << ", " << std::endl;
      }
      auto time  = entry2.first;
      auto value = entry2.second;
      if (time == ReplicaEqualAnalysisResults::initTime) {
        out << "initTime->" << value;
      } else {
        auto op = results.graphSchedules.at(tensor->getGraph().id).at(time);
        out << "  " << op->str() << "->" << value;
      }
      isFirst = false;
    }
    out << "]" << std::endl;
  }
  return out;
}

} // namespace popart
