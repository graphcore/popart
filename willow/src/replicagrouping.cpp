// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "popart/replicagrouping.hpp"

#include <tuple>
#include <vector>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/util/expressionchecking.hpp"

namespace popart {

ReplicaGrouping::ReplicaGrouping(unsigned numReplicas,
                                 unsigned stride,
                                 unsigned groupSize)
    : numReplicas_(numReplicas), stride_(stride), groupSize_(groupSize) {
  POPART_CHECK_NE(numReplicas, 0)
      << "The number of replicas in a `popart::ReplicaGrouping` must be a "
         "positive integer.";
  POPART_CHECK_NE(stride, 0)
      << "The stride in a `popart::ReplicaGrouping` must be a positive "
         "integer.";
  POPART_CHECK_NE(groupSize, 0)
      << "The group size in a `popart::ReplicaGrouping` must be a positive "
         "integer.";

  POPART_CHECK_EQ(numReplicas % (stride * groupSize), 0)
      << "The number of replicas in a `popart::ReplicaGrouping` must be "
         "divisible by the product of the stride and the group size.";

  if (groupSize == 1 && stride != 1) {
    stride_ = 1;
  }
}

ReplicaGrouping::ReplicaGrouping(unsigned numReplicas)
    : ReplicaGrouping(numReplicas, 1, numReplicas) {}

unsigned ReplicaGrouping::getNumReplicas() const { return numReplicas_; }

unsigned ReplicaGrouping::getStride() const { return stride_; }

unsigned ReplicaGrouping::getGroupSize() const { return groupSize_; }

unsigned ReplicaGrouping::getNumGroups() const {
  return numReplicas_ / groupSize_;
}

unsigned ReplicaGrouping::getGroupAt(unsigned replica) const {
  checkReplicaIsValid(replica);

  unsigned group = replica % stride_;
  group += stride_ * (replica / (stride_ * groupSize_));
  return group;
}

unsigned ReplicaGrouping::getIndexInGroupAt(unsigned replica) const {
  checkReplicaIsValid(replica);

  return (replica - getReplicaAt(getGroupAt(replica))) / stride_;
}

unsigned ReplicaGrouping::getReplicaAt(unsigned group, unsigned index) const {
  checkGroupIsValid(group);
  checkIndexInGroupIsValid(index);

  unsigned replica =
      (group / stride_) * (stride_ * groupSize_) + group % stride_;
  replica += index * stride_;
  return replica;
}

std::vector<unsigned> ReplicaGrouping::getReplicasAt(unsigned group) const {
  checkGroupIsValid(group);

  std::vector<unsigned> result;
  result.reserve(groupSize_);

  unsigned firstReplica = getReplicaAt(group);
  for (unsigned i = 0; i < groupSize_; i++) {
    result.push_back(firstReplica + i * stride_);
  }

  return result;
}

ReplicaGrouping ReplicaGrouping::getTranspose() const {
  auto numGroups = getNumGroups();

  if (stride_ == 1) {
    unsigned newStride    = groupSize_;
    unsigned newGroupSize = numGroups;
    return {numReplicas_, newStride, newGroupSize};
  }
  if (groupSize_ * stride_ == numReplicas_) {
    unsigned newStride    = 1;
    unsigned newGroupSize = stride_;
    return {numReplicas_, newStride, newGroupSize};
  }
  throw error("The transpose of '{}' cannot be represented as a "
              "`popart::ReplicaGrouping`.",
              str());
}

std::string ReplicaGrouping::str() const {
  return logging::format(
      "ReplicaGrouping(numReplicas={}, stride={}, groupSize={})",
      getNumReplicas(),
      stride_,
      groupSize_);
}

bool ReplicaGrouping::operator==(const ReplicaGrouping &other) const {
  return std::tie(numReplicas_, stride_, groupSize_) ==
         std::tie(other.numReplicas_, other.stride_, other.groupSize_);
}

bool ReplicaGrouping::operator!=(const ReplicaGrouping &other) const {
  return !operator==(other);
}

void ReplicaGrouping::checkReplicaIsValid(const unsigned replica) const {
  POPART_CHECK_LT(replica, getNumReplicas())
      << "The requested replica index is outside the valid range for "
      << "'" << str() << "'.";
}

void ReplicaGrouping::checkGroupIsValid(const unsigned group) const {
  POPART_CHECK_LT(group, getNumGroups())
      << "The requested group index is outside the valid range for "
      << "'" << str() << "'.";
}

void ReplicaGrouping::checkIndexInGroupIsValid(const unsigned index) const {
  POPART_CHECK_LT(index, getGroupSize())
      << "The requested index is outside the valid range for "
      << "'" << str() << "'.";
}

std::ostream &operator<<(std::ostream &os, const ReplicaGrouping &grouping) {
  os << grouping.str();
  return os;
}

} // namespace popart
