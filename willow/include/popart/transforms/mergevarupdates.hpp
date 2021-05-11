// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MERGECONSTSGDVARUPDATES_HPP
#define GUARD_NEURALNET_MERGECONSTSGDVARUPDATES_HPP

#include <map>
#include <string>
#include <popart/op/varupdate.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

class VarUpdateOp;

struct VarUpdateStartEnd {
public:
  using Start = int64_t;
  using End   = int64_t;
  VarUpdateStartEnd(VarUpdateOp *v, Start s, End e)
      : vop(v), start(s), end(e) {}
  VarUpdateOp *vop;
  Start start;
  End end;

  bool operator<(const VarUpdateStartEnd &rhs) const {
    return std::make_tuple(vop, start, end) <
           std::make_tuple(rhs.vop, rhs.start, rhs.end);
  }
};

class MergeVarUpdates : public Transform {
public:
  // A unique identifier identifiying a group of VarUpdates which can be merged.
  // They must have the sam learning rate, type, etc. Two VarUpdates with the
  // same PartitionId can be merged
  using PartitionId = std::string;
  PartitionId getPartitionId(Op *op) const;

  using PartitionMap = std::map<PartitionId, std::vector<VarUpdateStartEnd>>;

  // Transform the Graph by merging VarUpydates returned by getFinal
  virtual bool apply(Graph &) const final;

  // partition all VarUpdateOps in the the largest possible groups
  PartitionMap getLargestGroupTargetsMap(const Graph &) const;

private:
  virtual PartitionMap getFinal(const Graph &) const = 0;
};

// A Transformation to merge all Ops which inherit from VarUpdateOp into as few
// groups as possible
class MergeAllVarUpdates : public MergeVarUpdates {
public:
  static std::size_t id();
  MergeAllVarUpdates() : MergeVarUpdates() {}
  virtual ~MergeAllVarUpdates() override {}
  virtual std::size_t getId() const final { return id(); }
  virtual std::string getName() const final { return "MergeAllVarUpdates"; }

private:
  PartitionMap getFinal(const Graph &graph) const final {
    return getLargestGroupTargetsMap(graph);
  }
};

class MergeAuto : public MergeVarUpdates {
public:
  int64_t getThresholdMemory(const Graph &) const;
};

// Reshape (really just a flatten to {1, nelms}), Slice, Concat, so that there
// are as many VarUpdates as possible of size EXACTLY mergeVarUpdateMemThreshold
class MergeTightThreshold : public MergeAuto {
public:
  static std::size_t id();
  MergeTightThreshold() : MergeAuto() {}
  virtual ~MergeTightThreshold() override {}
  virtual std::size_t getId() const final { return id(); }
  virtual std::string getName() const final { return "MergeTightThreshold"; }

private:
  PartitionMap getFinal(const Graph &) const final;
};

// Reshape,  Concat, so that as many VarUpdates as possible of size
// as least mergeVarUpdateMemThreshold. That is, keep Concatenating Variables
// until the cumulative size is above mergeVarUpdateMemThreshold, then perform
// a VarUpdate on those concatenated Vars
class MergeLooseThreshold : public MergeAuto {
public:
  static std::size_t id();
  MergeLooseThreshold() : MergeAuto() {}
  virtual ~MergeLooseThreshold() override {}
  virtual std::size_t getId() const final { return id(); }
  virtual std::string getName() const final { return "MergeLooseThreshold"; }
  int64_t getMemToPlayWithAtPeak(const Graph &) const;

private:
  PartitionMap getFinal(const Graph &) const final;
};

} // namespace popart

#endif
