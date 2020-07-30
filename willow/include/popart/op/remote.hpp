// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REMOTE_HPP
#define GUARD_NEURALNET_REMOTE_HPP

#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {

class RemoteStoreOp : public Op {
public:
  RemoteStoreOp(const OperatorIdentifier &,
                const Op::Settings &,
                RemoteBufferId rbid_ = -1UL);

  std::unique_ptr<Op> clone() const final;
  void setup() final {}

  static InIndex getRemoteBufferOffsetInIndex() { return 1; }
  static InIndex getLocalTensorInIndex() { return 0; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  bool isOutlineable() const final { return true; }
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  void setRemoteBufferId(RemoteBufferId remotebuffer_id_) {
    remotebuffer_id = remotebuffer_id_;
  }
  RemoteBufferId getRemoteBufferId() const { return remotebuffer_id; }

private:
  RemoteBufferId remotebuffer_id;
};

class RemoteLoadOp : public Op {
public:
  RemoteLoadOp(const OperatorIdentifier &,
               const Op::Settings &,
               RemoteBufferId rbid_ = -1UL);

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  static InIndex getRemoteBufferOffsetInIndex() { return 1; }
  static OutIndex getLocalTensorInIndex() { return 0; }
  static OutIndex getLocalTensorOutIndex() { return 0; }

  view::Regions modifies(InIndex) const final;
  view::Regions aliases(InIndex, OutIndex) const final;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  bool isOutlineable() const final { return true; }
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  void setRemoteBufferId(RemoteBufferId remotebuffer_id_) {
    remotebuffer_id = remotebuffer_id_;
  }
  RemoteBufferId getRemoteBufferId() const { return remotebuffer_id; }

private:
  RemoteBufferId remotebuffer_id;
};

} // namespace popart

#endif
