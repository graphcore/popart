// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_IPUCOPY_HPP
#define GUARD_NEURALNET_IPUCOPY_HPP

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>

namespace popart {

using SourceIpuMap    = std::map<TensorId, uint64_t>;
using SourceTensorMap = std::map<uint64_t, std::vector<TensorId>>;

class IpuCopyOp : public Op {
public:
  IpuCopyOp(const OperatorIdentifier &_opid,
            uint64_t _destIpu,
            const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  uint64_t getDestIpu() const { return destIpu; }
  const SourceIpuMap &getSourceIpus() const;
  const SourceTensorMap &getSourceTensors() const;
  uint64_t getSourceIpu(const TensorId &tenId) const;
  uint64_t getSourceIpu() const;
  uint64_t getMinSourceIpu() const;
  uint64_t getMaxSourceIpu() const;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  // T9357: Phased execution needs this set to true.
  // T13645: Investigate if this can be problematic.
  // Should be session option.
  bool isOutlineable() const override;

  bool isIpuCopyOp() const final;

  bool copiesOptimizerTensors() const final;

  void connectInTensor(InIndex, TensorId, uint64_t sourceIpu);

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
                                   std::set<OpId> visited = {}) const override;
  VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex index,
                                    std::set<OpId> visited = {}) const override;

private:
  void connectInTensor(InIndex, TensorId) override {
    throw error(
        "Must supply a sourceIpu when calling "
        "IpuCopyOp::connectInTensor(InIndex, TensorId, uint64_t sourceIpu)");
  }

  SourceIpuMap sourceIpus;
  SourceTensorMap sourceTensors;
  uint64_t destIpu;
};
} // namespace popart

#endif
