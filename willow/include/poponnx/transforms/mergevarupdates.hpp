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

protected:
  // partition all VarUpdateOps by PartitionId
  std::map<PartitionId, std::vector<VarUpdateOp *>>
  getLargestGroupTargetsMap(const Graph &graph) const;
};

// A Transformation to merge all Ops which inherit from VarUpdateOp into as few
// groups as possible
class MergeAllVarUpdates : public MergeVarUpdates {
public:
  static std::size_t id();
  MergeAllVarUpdates() : MergeVarUpdates() {}
  virtual ~MergeAllVarUpdates() override {}
  virtual bool apply(Graph &graph) const override final;
  virtual std::size_t getId() const override final { return id(); }
  virtual std::string getName() const override final {
    return "MergeAllVarUpdates";
  }
};

// TODO T8703: Auto to inherit from MergeVarUpdates

} // namespace poponnx

#endif
