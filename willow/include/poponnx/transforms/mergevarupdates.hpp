#ifndef GUARD_NEURALNET_MERGECONSTSGDVARUPDATES_HPP
#define GUARD_NEURALNET_MERGECONSTSGDVARUPDATES_HPP

#include <boost/optional/optional_io.hpp>
#include <map>
#include <string>
#include <poponnx/op/varupdate.hpp>
#include <poponnx/transforms/transform.hpp>

namespace poponnx {

class VarUpdateOp;

class MergeVarUpdates : public Transform {
public:
  // A unique identifier identifiying partitions of VarUpdates
  // which can be merged. They must have the sam learning rate, etc.,
  // captured in the std::string argument of the tuple.
  using PartitionId = std::string;
  PartitionId getPartitionId(Op *op) const;
  using VarUpdatePartition = std::map<PartitionId, std::vector<VarUpdateOp *>>;

  virtual VarUpdatePartition getGroupTargetsMap(const Graph &) const = 0;

  virtual bool apply(Graph &) const override final;

protected:
  // partition all VarUpdateOps by PartitionId
  VarUpdatePartition getLargestGroupTargetsMap(const Graph &) const;
};

// A Transformation to merge all Ops which inherit from VarUpdateOp into as few
// groups as possible
class MergeAllVarUpdates : public MergeVarUpdates {
public:
  static std::size_t id();
  MergeAllVarUpdates() : MergeVarUpdates() {}
  virtual ~MergeAllVarUpdates() override {}
  virtual std::size_t getId() const override final { return id(); }
  virtual std::string getName() const override final {
    return "MergeAllVarUpdates";
  }
  VarUpdatePartition getGroupTargetsMap(const Graph &graph) const final {
    return getLargestGroupTargetsMap(graph);
  }
};

class MergeAutoVarUpdates : public MergeVarUpdates {
public:
  static std::size_t id();
  MergeAutoVarUpdates() : MergeVarUpdates() {}
  virtual ~MergeAutoVarUpdates() override {}
  virtual std::size_t getId() const override final { return id(); }
  virtual std::string getName() const override final {
    return "MergeAutoVarUpdates";
  }
  VarUpdatePartition getGroupTargetsMap(const Graph &) const final;
};

} // namespace poponnx

#endif
