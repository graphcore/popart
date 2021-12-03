// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REMOTE_HPP
#define GUARD_NEURALNET_REMOTE_HPP

#include <popart/op.hpp>
#include <popart/op/exchange/remotebase.hpp>

namespace popart {

/**
 * Remote Store Op
 *
 * Stores a tensor to a remote (off-chip) buffer.
 * This \c Op is typically used when the user wants to store several different
 * identically shaped tensors to the same remote buffer by specifying the
 * \c offset (see below).
 *
 * This class takes between one and two \c TensorIds as inputs
 * (as indicated in \see opidentifier.hpp).
 *
 * 1. The \c TensorId of the \c inTensor to copy to remote memory.
 * 2. The (optional) \c TensorId 0-rank tensor called \c offset .
 *    - If set to a value >= 0 \c offset will specify the row in the remote
 *      buffer the \c inTensor will be written to (see below for explanation).
 *    - If set to -1 \c RemoteSetup will assign a unique value.
 *
 * If \c inTensor is of rank \a x , the remote buffer of a certain
 * \c RemoteBufferId will be of rank \a x+1, where the new dimension (the row)
 * will be of size \c N.
 *
 * \c Op instances with matching \c RemoteBufferId will \a outline together,
 * meaning that if multiple different tensors are to be stored under the same
 * remote buffer ID, a different offset value has to be supplied for each
 * tensor.
 *
 * For using the automatic \see RemoteSetup configuration, the \c offset
 * tensor should be a unique constant tensor per \c inTensor per
 * \c RemoteBufferId.
 * If the constant \c offset tensor has value -1, \c RemoteSetup will assign
 * a unique value, otherwise the supplied \c offset value will be used.
 * \c RemoteSetup will call \c Ir::setRemoteBufferInfo to configure the
 * shape (equal to the \c inTensor shape) and number of rows ( \c N ) in the
 * remote memory.
 *
 * If not using the automatic \c RemoteSetup, all \c offsets and
 * \c RemoteBufferIds need to be >= 0.
 * Each remote buffer ID needs then to be registered with
 * \c Ir::setRemoteBufferInfo manually.
 *
 * This Op does not have any output.
 *
 * See also \see RemoteLoadOp.
 **/
class RemoteStoreOp : public RemoteBaseOp {
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

  bool hasSideEffect() const override { return true; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;

  ExchangeDescriptor getExchangeDescriptor(int index) const final;
};

/**
 * Remote Load Op
 *
 * Loads a tensor from remote (off-chip) buffer.
 * The tensor will be loaded from the memory location corresponding to
 * \c RemoteBufferId, and will be stored in the memory location corresponding to
 * \c inTensor.
 *
 * This class takes between one and two \c TensorIds as inputs
 * (as indicated in \see opidentifier.hpp).
 *
 * 1. The \c TensorId of the \c inTensor.
 *    - In the \a inplace version this will be aliased to the output tensor
 *    - In the \a outplace version this \c Op will clone the \c inTensor, then
 *      write the loaded data to the clone
 * 2. The (optional) \c TensorId to a 0-rank tensor called \c offset .
 *    - If set to a value >= 0 \c offset will specify the row in the remote
 *      buffer the tensor will be loaded.
 *    - If set to -1 \c RemoteSetup will assign a unique value.
 *
 * The relationship between \c offset, \c RemoteBufferId and \c RemoteSetup
 * is thoroughly described in \see RemoteStoreOp.
 *
 * The output is the \c TensorId of the loaded tensor.
 *
 * See also \see RemoteStoreOp.
 **/
class RemoteLoadOp : public RemoteBaseOp {
public:
  /**
   * Construct the \c RemoteLoadOp
   *
   * Parameters specifically related to this class:
   *
   * \param RemoteBufferId The id of the remote buffer.
   *                       Can be any integer.
   *                       If not specified (or set to -1), the \c RemoteSetup
   *                       will automatically choose the right buffer.
   *                       The \c RemoteBufferId can only be used with tensors
   *                       of identical shape.
   *
   * See constructor of the parent class for the rest of input parameters.
   **/
  RemoteLoadOp(const OperatorIdentifier &,
               const Op::Settings &,
               RemoteBufferId rbid_ = -1UL);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  static OutIndex getLocalTensorOutIndex() { return 0; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const final;

  ExchangeDescriptor getExchangeDescriptor(int index) const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const override;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const override;

  void growAliasModel(AliasModel &) const final;

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &, OperatorIdentifier) const final;
};

/**
 * Remote Load Inplace Op
 *
 * See also \see RemoteLoadOp for explanation.
 **/
class RemoteLoadInplaceOp : public RemoteLoadOp {
public:
  /**
   * Construct the \c RemoteLoadInplaceOp
   *
   * See constructor of the parent class for the input parameters.
   **/
  RemoteLoadInplaceOp(const OperatorIdentifier &,
                      const Op::Settings &,
                      RemoteBufferId rbid_ = -1UL);
  RemoteLoadInplaceOp(const RemoteLoadOp &);

  std::unique_ptr<Op> clone() const final;

  view::Regions modifies(InIndex) const final;
  view::Regions aliases(InIndex, OutIndex) const final;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final;

  ExchangeDescriptor getExchangeDescriptor(int index) const final;
};

} // namespace popart

#endif
