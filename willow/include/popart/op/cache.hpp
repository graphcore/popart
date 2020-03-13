// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CACHE_HPP
#define GUARD_NEURALNET_CACHE_HPP

#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {

class CacheStoreOp : public Op {
public:
  CacheStoreOp(const OperatorIdentifier &,
               const Op::Settings &,
               RemoteBufferId id = -1UL);

  std::unique_ptr<Op> clone() const final;
  void setup() final {}

  static InIndex getRemoteBufferOffsetInIndex() { return 1; }
  static InIndex getCachedTensorInIndex() { return 0; }

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

class CacheLoadOp : public Op {
public:
  CacheLoadOp(const OperatorIdentifier &,
              const Op::Settings &,
              RemoteBufferId id = -1UL);

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  static InIndex getRemoteBufferOffsetInIndex() { return 1; }
  static OutIndex getCachedTensorInIndex() { return 0; }
  static OutIndex getCachedTensorOutIndex() { return 0; }

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
