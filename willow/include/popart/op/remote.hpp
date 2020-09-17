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

  bool hasSideEffect() const override { return true; }

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

// The inputs to a RemoteExchangeOp (N loads, M stores) are:
// [load-data0, ..., load-dataN, store-data0, ..., store-dataM]
// [arg-load-data0, ..., arg-load-dataN, arg-store-data0, ..., arg-store-dataM]
// The inputs to a remote exchange op are:
// [load-data0, ..., load-dataN]
class RemoteExchangeOp : public Op {
public:
  RemoteExchangeOp(const OperatorIdentifier &,
                   const Op::Settings &,
                   const std::vector<RemoteBufferId>,
                   const std::vector<std::pair<OptionalVGraphId, TileSet>>);

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  view::Regions modifies(InIndex) const final;
  view::Regions aliases(InIndex, OutIndex) const final;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  bool isOutlineable() const final { return true; }
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  int numLoads() const;
  int numStores() const;

  void setRemoteBufferId(InIndex index, RemoteBufferId remotebuffer_id) {
    remotebufferIds[index] = remotebuffer_id;
  }
  RemoteBufferId getRemoteBufferId(InIndex index) const {
    return remotebufferIds.at(index);
  }

  VGraphIdAndTileSet getIntrospectionInVirtualGraphId(InIndex) const final;
  VGraphIdAndTileSet getIntrospectionOutVirtualGraphId(OutIndex) const final;

private:
  std::vector<RemoteBufferId> remotebufferIds;
  std::vector<std::pair<OptionalVGraphId, TileSet>> vgidAndTiles;
};

} // namespace popart

#endif
