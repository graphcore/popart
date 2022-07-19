// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_EXCHANGE_HOSTBASE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_EXCHANGE_HOSTBASE_HPP_

#include <memory>
#include <popart/op.hpp>
#include <popart/op/exchange/exchange.hpp>

#include "popart/names.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class HostBaseOp : public ExchangeBaseOp {
public:
  HostBaseOp(const OperatorIdentifier &_opid,
             const Op::Settings &settings_,
             TensorId sid_)
      : ExchangeBaseOp(_opid, settings_), hostStreamTensorId(sid_) {}

  static InIndex getLocalTensorInIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool canShard() const final { return false; }

  std::unique_ptr<Op> clone() const override = 0;

  bool hasSideEffect() const override { return true; }

  void setHostStreamTensorId(TensorId stream_id_) {
    hostStreamTensorId = stream_id_;
  }
  TensorId getHostStreamTensorId() const { return hostStreamTensorId; }

private:
  TensorId hostStreamTensorId;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_EXCHANGE_HOSTBASE_HPP_
