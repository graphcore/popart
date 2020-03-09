#ifndef GUARD_NEURALNET_DYNAMICUPDATE_HPP
#define GUARD_NEURALNET_DYNAMICUPDATE_HPP

#include <popart/op/dynamic/dynamicbase.hpp>

namespace popart {

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
