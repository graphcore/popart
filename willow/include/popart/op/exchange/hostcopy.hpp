// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_HOSTCOPY_HPP
#define GUARD_NEURALNET_HOSTCOPY_HPP

#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/op/exchange/exchange.hpp>

namespace popart {

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
class HostLoadOp : public ExchangeBaseOp {
public:
  HostLoadOp(const OperatorIdentifier &, const Op::Settings &, TensorId sid_);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  static InIndex getLocalTensorInIndex() { return 0; }
  static OutIndex getLocalTensorOutIndex() { return 0; }

  view::Regions modifies(InIndex) const override;
  view::Regions aliases(InIndex, OutIndex) const override;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  virtual std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
  fwdPropagateIsReplicaEqual(const AliasModel &aliasModel,
                             const ReplEqInputMap &inputMap,
                             ReplicaEqualAnalysisProxy &proxy) const override;

  void growAliasModel(AliasModel &m) const override { growAliasModelMulti(m); }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  void setHostStreamTensorId(TensorId stream_id_) {
    hostStreamTensorId = stream_id_;
  }
  TensorId getHostStreamTensorId() const { return hostStreamTensorId; }

  bool canShard() const final { return false; }

  ExchangeDescriptor getExchangeDescriptor(int index) const final;

private:
  TensorId hostStreamTensorId;
};

class HostStoreOp : public ExchangeBaseOp {
public:
  HostStoreOp(const OperatorIdentifier &, const Op::Settings &, TensorId sid_);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  static InIndex getLocalTensorInIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  void setHostStreamTensorId(TensorId stream_id_) {
    hostStreamTensorId = stream_id_;
  }
  TensorId getHostStreamTensorId() const { return hostStreamTensorId; }

  bool canShard() const final { return false; }

  bool hasSideEffect() const override { return true; }

  ExchangeDescriptor getExchangeDescriptor(int index) const final;

private:
  TensorId hostStreamTensorId;
};
} // namespace popart

#endif
