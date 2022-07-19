// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_EXCHANGE_HOSTCOPY_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_EXCHANGE_HOSTCOPY_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <poprithms/memory/inplace/proposal.hpp>
#include <popart/op.hpp>
#include <popart/op/exchange/exchange.hpp>
#include <popart/op/exchange/hostbase.hpp>

#include "popart/names.hpp"

namespace popart {
class AliasModel;
class ReplicaEqualAnalysisProxy;
struct OperatorIdentifier;

/**
 * Host Load Op: an op to represent the transfer of data from the host to the
 * device. It uses the existing host to device transfers created when building
 * the IR, but defers the actual poplar::Copy until the op itself runs. This
 * allows the copy to be scheduled as part of the normal op scheduling.
 *
 * There is a stage in the IR which adds the following ops:
 *
 * Device :: InitOp -> input_prehostload -> HostLoadOp -> input -> etc...
 *                                                              /
 * Host   ::                                          data -> stream
 */
class HostLoadOp : public HostBaseOp {
public:
  HostLoadOp(const OperatorIdentifier &, const Op::Settings &, TensorId sid_);

  std::unique_ptr<Op> clone() const override;
  void setup() override;

  static OutIndex getLocalTensorOutIndex() { return 0; }

  virtual std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
  fwdPropagateIsReplicaEqual(const AliasModel &aliasModel,
                             const ReplEqInputMap &inputMap,
                             ReplicaEqualAnalysisProxy &proxy) const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const override;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const override;

  void growAliasModel(AliasModel &m) const final;

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &, OperatorIdentifier) const final;

  ExchangeDescriptor getExchangeDescriptor(int index) const override;
};

class HostLoadInplaceOp : public HostLoadOp {
public:
  HostLoadInplaceOp(const OperatorIdentifier &,
                    const Op::Settings &,
                    TensorId sid_);
  HostLoadInplaceOp(const HostLoadOp &);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  view::Regions modifies(InIndex) const override;
  view::Regions aliases(InIndex, OutIndex) const override;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final;

  ExchangeDescriptor getExchangeDescriptor(int index) const final;
};

class HostStoreOp : public HostBaseOp {
public:
  HostStoreOp(const OperatorIdentifier &, const Op::Settings &, TensorId sid_);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  ExchangeDescriptor getExchangeDescriptor(int index) const final;
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_EXCHANGE_HOSTCOPY_HPP_
