// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DYNAMICUPDATE_HPP
#define GUARD_NEURALNET_DYNAMICUPDATE_HPP

#include <popart/op/dynamic/dynamicbase.hpp>

namespace popart {

/**
 * Dynamic Update Op
 *
 * This class takes three \c TensorIds as input (as indicated in \see
 * graphcoreoperators.hpp)
 *
 * 1. The \c TensorId of the tensor to be updated.
 * 2. The \c TensorId of the index of the starting point of the slice
 *    ( \see DynamicBaseOp for explanation).
 * 3. The \c TensorId to update with (must match dimension with
 *    ( \c index, \c axes, \c sizes )).
 *
 * The output is the \c TensorId of the updated tensor.
 *
 * \see DynamicTernaryBaseOp for details.
 **/
class DynamicUpdateOp : public DynamicTernaryBaseOp {
public:
  DynamicUpdateOp(const OperatorIdentifier &_opid,
                  std::vector<int64_t> axes_,
                  std::vector<int64_t> sizes_,
                  bool noOverlap_,
                  const Op::Settings &settings_,
                  TensorInfo updateInInfo_ = TensorInfo());
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const override;
};

class DynamicUpdateInplaceOp : public DynamicTernaryBaseInplaceOp {
public:
  DynamicUpdateInplaceOp(const DynamicUpdateOp &dynamicUpdateOp);
  DynamicUpdateInplaceOp(const OperatorIdentifier &_opid,
                         std::vector<int64_t> axes_,
                         std::vector<int64_t> sizes_,
                         bool noOverlap_,
                         const Op::Settings &settings_,
                         TensorInfo updateInInfo_ = TensorInfo());
  std::unique_ptr<Op> clone() const final;
};

class DynamicUpdateToUpdateGradOp : public DynamicBinaryBaseOp {
public:
  DynamicUpdateToUpdateGradOp(const OperatorIdentifier &_opid,
                              std::vector<int64_t> axes_,
                              std::vector<int64_t> sizes_,
                              bool noOverlap_,
                              const Op::Settings &);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

class DynamicUpdateUpdaterGradOp : public DynamicSliceBaseOp {
public:
  DynamicUpdateUpdaterGradOp(const OperatorIdentifier &_opid,
                             std::vector<int64_t> axes_,
                             std::vector<int64_t> sizes_,
                             bool noOverlap_,
                             const Op::Settings &);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace popart

#endif
