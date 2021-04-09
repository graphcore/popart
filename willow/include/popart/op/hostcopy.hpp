// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_HOSTCOPY_HPP
#define GUARD_NEURALNET_HOSTCOPY_HPP

#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>

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
class HostLoadOp : public Op {
public:
  HostLoadOp(const OperatorIdentifier &,
             const Op::Settings &,
             HostStreamId hsid_ = std::numeric_limits<uint64_t>::max() - 1);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  static InIndex getLocalTensorInIndex() { return 0; }
  static OutIndex getLocalTensorOutIndex() { return 0; }

  view::Regions modifies(InIndex) const override;
  view::Regions aliases(InIndex, OutIndex) const override;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  bool isOutlineable() const final { return true; }
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  void setStreamId(HostStreamId stream_id_) { stream_id = stream_id_; }
  HostStreamId getHostStreamId() const { return stream_id; }

  bool canShard() const final { return false; }

private:
  HostStreamId stream_id;
};

class HostStoreOp : public Op {
public:
  HostStoreOp(const OperatorIdentifier &,
              const Op::Settings &,
              HostStreamId hsid_ = std::numeric_limits<uint64_t>::max() - 1);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  static InIndex getLocalTensorInIndex() { return 0; }

  view::Regions modifies(InIndex) const override;
  view::Regions aliases(InIndex, OutIndex) const override;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  bool isOutlineable() const final { return true; }
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  void setStreamId(HostStreamId stream_id_) { stream_id = stream_id_; }
  HostStreamId getHostStreamId() const { return stream_id; }

  bool canShard() const final { return false; }

private:
  HostStreamId stream_id;
};
} // namespace popart

#endif
