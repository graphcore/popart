// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_IPUCOPY_HPP
#define GUARD_NEURALNET_IPUCOPY_HPP

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>

namespace popart {

using SourceIpuMap    = std::map<TensorId, VGraphId>;
using SourceTensorMap = std::map<VGraphId, std::vector<TensorId>>;

class IpuCopyOp : public Op {
public:
  IpuCopyOp(const OperatorIdentifier &_opid,
            VGraphId _destIpu,
            const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  VGraphId getDestIpu() const { return destIpu; }
  const SourceIpuMap &getSourceIpus() const;
  const SourceTensorMap &getSourceTensors() const;
  VGraphId getSourceIpu(const TensorId &tenId) const;
  VGraphId getSourceIpu() const;
  VGraphId getMinSourceIpu() const;
  VGraphId getMaxSourceIpu() const;

  void setSourceIpus(const SourceIpuMap sourceIpus);
  void setSourceTensors(const SourceTensorMap sourceTensors);

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  // T9357: Phased execution needs this set to true.
  // T13645: Investigate if this can be problematic.
  // Should be session option.
  bool isOutlineable() const override;

  bool isIpuCopyOp() const final;

  bool copiesOptimizerTensors() const final;

  void connectInTensor(InIndex, TensorId, VGraphId sourceIpu) override;

  // A string of the form "[ sourceIpus ] --> [ destIpu ]"
  std::string getFromToStr() const;

  void disconnectInTensor(InIndex, Tensor *) override;

  bool canShard() const override {
    // For batch serialization, IpuCopyOp signifies changing virtual graphs
    return !getGraph()
                .getIr()
                .getSessionOptions()
                .batchSerializationSettings.concatOnVirtualGraphChange;
  }

  VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(InIndex index,
                                   std::set<OpId> &visited) const override;
  VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex index,
                                    std::set<OpId> &visited) const override;

private:
  void connectInTensor(InIndex, TensorId) override {
    throw error(
        "Must supply a sourceIpu when calling "
        "IpuCopyOp::connectInTensor(InIndex, TensorId, uint64_t sourceIpu)");
  }

  SourceIpuMap sourceIpus;
  SourceTensorMap sourceTensors;
  VGraphId destIpu;
};

} // namespace popart

#endif
