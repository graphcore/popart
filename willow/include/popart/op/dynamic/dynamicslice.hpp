// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DYNAMICSLICE_HPP
#define GUARD_NEURALNET_DYNAMICSLICE_HPP

#include <popart/op/dynamic/dynamicbase.hpp>

namespace popart {

/**
 * Dynamic Slice Op
 *
 * This Op takes two or three \c TensorIds as input (as indicated in \see
 * graphcoreoperators.hpp)
 *
 * 1. The \c TensorId of tensor to slice from.
 * 2. The (optional) \c TensorId of the index of the starting point of the slice
 *    ( \see DynamicBaseOp for explanation).
 * 3. The \c TensorId of the tensor to write the slice into (not used in
 *    outplace variant).
 *
 * The output is the \c TensorId of the sliced tensor.
 **/
class DynamicSliceOp : public DynamicSliceBaseOp {
public:
  DynamicSliceOp(const OperatorIdentifier &_opid,
                 std::vector<int64_t> axes_,
                 std::vector<int64_t> sizes_,
                 bool noOverlap_,
                 const Op::Settings &);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  static InIndex getSliceInIndex() { return 2; }

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const override;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const override;

  void growAliasModel(AliasModel &) const override;

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &, OperatorIdentifier) const override;
};

/**
 * Dynamic Slice Inplace Op
 *
 * This Op takes two or three \c TensorIds as input (as indicated in \see
 * graphcoreoperators.hpp)
 *
 * 1. The \c TensorId of tensor to slice from.
 * 2. The (optional) \c TensorId of the index of the starting point of the slice
 *    ( \see DynamicBaseOp for explanation).
 * 3. The \c TensorId of the tensor to write the slice into (not used in
 *    outplace variant).
 *
 * The output is the \c TensorId of the sliced tensor, aliased
 **/
class DynamicSliceInplaceOp : public DynamicSliceOp {
public:
  DynamicSliceInplaceOp(const OperatorIdentifier &_opid,
                        std::vector<int64_t> axes_,
                        std::vector<int64_t> sizes_,
                        bool noOverlap_,
                        const Op::Settings &);
  DynamicSliceInplaceOp(const DynamicSliceOp &);
  std::unique_ptr<Op> clone() const final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  view::Regions modifies(InIndex) const override;
  view::Regions aliases(InIndex, OutIndex) const override;
};

class DynamicSlicePadGradOp : public DynamicBaseOp {
public:
  DynamicSlicePadGradOp(const OperatorIdentifier &_opid,
                        std::vector<int64_t> axes_,
                        std::vector<int64_t> sizes_,
                        bool noOverlap_,
                        const Op::Settings &settings_,
                        TensorInfo updateInInfo_ = TensorInfo());
  void setup() final;
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const override;
  static InIndex getInIndex() { return 2; }
  std::set<InIndex> optionalInputs() const override { return {0}; }

protected:
  TensorInfo updateInInfo;
};

} // namespace popart

#endif
