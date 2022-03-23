// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_HOSTBASE_HPP
#define GUARD_NEURALNET_HOSTBASE_HPP

#include <popart/op.hpp>
#include <popart/op/exchange/exchange.hpp>

namespace popart {

class HostBaseOp : public ExchangeBaseOp {
public:
  HostBaseOp(const OperatorIdentifier &_opid,
             const Op::Settings &settings_,
             TensorId sid_)
      : ExchangeBaseOp(_opid, settings_), hostStreamTensorId(sid_){};

  static InIndex getLocalTensorInIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool canShard() const final { return false; }

  bool hasSideEffect() const override { return true; }

  void setHostStreamTensorId(TensorId stream_id_) {
    hostStreamTensorId = stream_id_;
  }
  TensorId getHostStreamTensorId() const { return hostStreamTensorId; }

private:
  TensorId hostStreamTensorId;
};

} // namespace popart

#endif
