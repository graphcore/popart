// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REMOTEBASE_HPP
#define GUARD_NEURALNET_REMOTEBASE_HPP

#include <popart/op.hpp>
#include <popart/op/exchange/exchange.hpp>

namespace popart {

class RemoteBaseOp : public ExchangeBaseOp {
public:
  RemoteBaseOp(const OperatorIdentifier &_opid,
               const Op::Settings &settings_,
               RemoteBufferId rbid_)
      : ExchangeBaseOp(_opid, settings_), remoteBufferId(rbid_) {}
  std::unique_ptr<Op> clone() const = 0;

  virtual RemoteBufferId getRemoteBufferId() const final {
    return remoteBufferId;
  }

  static InIndex getLocalTensorInIndex() { return 0; }
  static InIndex getRemoteBufferOffsetInIndex() { return 1; }

  virtual bool canShard() const final { return true; }
  virtual void setRemoteBufferId(RemoteBufferId remoteBufferId_) final {
    remoteBufferId = remoteBufferId_;
  }

  virtual void appendOutlineAttributes(OpSerialiserBase &) const final;

protected:
  RemoteBufferId remoteBufferId;
};

} // namespace popart

#endif
