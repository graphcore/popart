// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REMOTE_HPP
#define GUARD_NEURALNET_REMOTE_HPP

#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/op/exchange/exchange.hpp>

namespace popart {

/**
 * Remote Store Op
 *
 * Stores \c inTensor to a remote (off-chip) buffer.
 * This \c Op is typically used when the user wants to store several different
 * identically shaped tensors to the same remote buffer by specifying the \c
 *offset \c Tensor (see below).
 *
 * This class takes between one and two \c Tensors as inputs
 * (as indicated in \see opidentifier.hpp).
 *
 * 1. The \c inTensor to copy to remote memory.
 * 2. The (optional) 0-rank \c Tensor called \c offset .
 *    - If set to a vaule >= 0 \c offset will specify the row in the remote
 *      buffer the \c Tensor will be written to (see below for explanation).
 *    - If set to -1 \c RemoteSetup will assign a unique value.
 *
 * If \c inTensor is of rank \a x , the remote buffer of a certain
 * \c RemoteBufferId will be of rank \a x+1, where the new dimension (the row)
 * will be of size \c N.
 *
 * \c Op instances with matching \c RemoteBufferIds will \a outline together,
 * meaning that if multiple different tensors are to be stored under the same
 * remote buffer IDs, a different offset value has to be supplied for each
 * tensor.
 *
 * For using the automatic \see RemoteSetup configuration, the \c offset
 * \c Tensor should be a unique constant tensor per \c inTensor per
 * \c RemoteBufferIds.
 * If the constant \c offset \c Tensor has value -1, \c RemoteSetup will assign
 * a unique value, otherwise the supplied \c offset value will be used.
 * \c RemoteSetup will call \c Ir::setRemoteBufferInfo to configure the
 * shape (equal to the \c inTensor shape) and number of rows ( \c N ) in the
 * remote memory.
 *
 * If not using the automatic \c RemoteSetup, all \c offsets and
 * \c RemoteBufferIds need to be >= 0.
 * Each remote buffer ID needs then to be registered with
 * \c Ir::setRemoteBufferInfo manually
 *
 *
 * This Op does not have any output.
 **/
class RemoteStoreOp : public ExchangeBaseOp {
public:
  /**
   * Construct the \c RemoteStoreOp
   *
   * Parameters specifically related to this class:
   *
   * \param RemoteBufferId The id of the remote buffer.
   *                       Can be any integer.
   *                       If not specified (or set to -1), the \c RemoteSetup
   *                       will automatically choose the right buffer.
   *                       The \c RemoteBufferIds can only be used with tensors
   *                       of identical shape.
   *
   * See constructor of the parent class for the rest of input parameters.
   **/
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
