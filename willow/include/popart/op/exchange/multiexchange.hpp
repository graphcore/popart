// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_EXCHANGE_MULTIEXCHANGE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_EXCHANGE_MULTIEXCHANGE_HPP_

#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/exchange/exchange.hpp>

#include "popart/names.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {
class AliasModel;
class OpSerialiserBase;
class ReplicaEqualAnalysisProxy;
struct OperatorIdentifier;

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

  virtual std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
  fwdPropagateIsReplicaEqual(const AliasModel &aliasModel,
                             const ReplEqInputMap &inputMap,
                             ReplicaEqualAnalysisProxy &proxy) const override;

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
                                   std::set<OpId> &visited) const final;
  VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex,
                                    std::set<OpId> &visited) const final;

  virtual void growAliasModel(AliasModel &m) const override {
    growAliasModelMulti(m);
  }

  bool canShard() const final { return false; }

  bool hasSideEffect() const final;

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;

  int getNumExchanges() const final { return descriptors.size(); }
  ExchangeDescriptor getExchangeDescriptor(int index) const final;

  /**
   * Map input index to a tuple of integers `(a,b)` that corresponds to the
   * input associated with `index`. That is, the `b`th input of
   * `getExchangeDescriptor(a)` corresponds to the input at `index`.
   *
   * \param index the input index to look up.
   * \return a pair of integers comprising the index of the descriptor and the
   * index of the input associated with the input within the descriptor.
   **/
  std::pair<int, int> inIndexToDescriptorIndex(InIndex index) const override;

  /**
   * Map output index to a tuple of integers `(a,b)` that corresponds to the
   * output associated with `index`. That is, the `b`th output of
   * `getExchangeDescriptor(a)` corresponds to the output at `index`.
   *
   * \param index the output index to look up.
   * \return a pair of integers comprising the index of the descriptor and the
   * index of the output associated with the output within the descriptor.
   **/
  std::pair<int, int> outIndexToDescriptorIndex(OutIndex index) const override;

  std::vector<InIndex> descriptorIndexToInIndices(int index) const override;
  std::vector<OutIndex> descriptorIndexToOutIndices(int index) const override;

private:
  ExchangeDescriptors descriptors;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_EXCHANGE_MULTIEXCHANGE_HPP_
