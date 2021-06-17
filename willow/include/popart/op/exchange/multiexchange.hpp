// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MULTIEXCHANGE_HPP
#define GUARD_NEURALNET_MULTIEXCHANGE_HPP

#include <popart/op.hpp>
#include <popart/op/exchange/exchange.hpp>

namespace popart {

std::ostream &operator<<(std::ostream &, const ExchangeDirection &);
std::ostream &operator<<(std::ostream &, const ExchangeDescriptor &);

using ExchangeDescriptors = std::vector<ExchangeDescriptor>;

class MultiExchangeOp : public ExchangeBaseOp {
public:
  MultiExchangeOp(const OperatorIdentifier &,
                  const Op::Settings &,
                  const std::vector<ExchangeDescriptor>);

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  view::Regions modifies(InIndex) const final;
  view::Regions aliases(InIndex, OutIndex) const final;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  int numLoads() const;
  int numStores() const;

  bool isRemote(int index) { return descriptors[index].isRemoteExchange(); }

  void setRemoteBufferId(int index, RemoteBufferId remotebuffer_id) {
    descriptors[index].setRemoteBufferId(remotebuffer_id);
  }

  RemoteBufferId getRemoteBufferId(int index) const {
    return descriptors.at(index).getRemoteBufferId();
  }

  VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(InIndex,
                                   std::set<OpId> visited = {}) const final;
  VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex,
                                    std::set<OpId> visited = {}) const final;

  virtual void growAliasModel(AliasModel &m) const override {
    growAliasModelMulti(m);
  }

  bool canShard() const final { return false; }

  bool hasSideEffect() const override { return numStores() > 0; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;

  int getNumExchanges() const final { return descriptors.size(); }
  ExchangeDescriptor getExchangeDescriptor(int index) const final;

  std::pair<int, int> inIndexToDescriptorIndex(InIndex index) const;
  std::pair<int, int> outIndexToDescriptorIndex(OutIndex index) const;

  std::vector<InIndex> descriptorIndexToInIndices(int index) const;
  std::vector<OutIndex> descriptorIndexToOutIndices(int index) const;

private:
  ExchangeDescriptors descriptors;
};

} // namespace popart

#endif
