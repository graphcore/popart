// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REMOTE_HPP
#define GUARD_NEURALNET_REMOTE_HPP

#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/op/exchange/exchange.hpp>

namespace popart {

class RemoteStoreOp : public ExchangeBaseOp {
public:
  RemoteStoreOp(const OperatorIdentifier &,
                const Op::Settings &,
                RemoteBufferId rbid_ = -1UL);

  std::unique_ptr<Op> clone() const final;
  void setup() final {}

  static InIndex getRemoteBufferOffsetInIndex() { return 1; }
  static InIndex getLocalTensorInIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  void setRemoteBufferId(RemoteBufferId remoteBufferId_) {
    remoteBufferId = remoteBufferId_;
  }
  RemoteBufferId getRemoteBufferId() const { return remoteBufferId; }

  bool hasSideEffect() const override { return true; }

  bool canShard() const final { return true; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;

  ExchangeDescriptor getExchangeDescriptor(int index) const final;

private:
  RemoteBufferId remoteBufferId;
};

class RemoteLoadOp : public ExchangeBaseOp {
public:
  RemoteLoadOp(const OperatorIdentifier &,
               const Op::Settings &,
               RemoteBufferId rbid_ = -1UL);

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  static InIndex getRemoteBufferOffsetInIndex() { return 1; }
  static InIndex getLocalTensorInIndex() { return 0; }
  static OutIndex getLocalTensorOutIndex() { return 0; }

  view::Regions modifies(InIndex) const final;
  view::Regions aliases(InIndex, OutIndex) const final;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  void setRemoteBufferId(RemoteBufferId remoteBufferId_) {
    remoteBufferId = remoteBufferId_;
  }
  RemoteBufferId getRemoteBufferId() const { return remoteBufferId; }

  bool canShard() const final { return true; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;

  virtual void growAliasModel(AliasModel &m) const override {
    growAliasModelMulti(m);
  }

  ExchangeDescriptor getExchangeDescriptor(int index) const final;

private:
  RemoteBufferId remoteBufferId;
};

} // namespace popart

#endif
