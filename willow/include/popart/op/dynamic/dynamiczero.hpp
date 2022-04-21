// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DYNAMICZERO_HPP
#define GUARD_NEURALNET_DYNAMICZERO_HPP

#include <popart/op/dynamic/dynamicbase.hpp>

namespace popart {

class DynamicZeroOp : public DynamicBinaryBaseOp {
public:
  DynamicZeroOp(const OperatorIdentifier &_opid,
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

class DynamicZeroGradOp : public DynamicBinaryBaseOp {
public:
  DynamicZeroGradOp(const OperatorIdentifier &_opid,
                    std::vector<int64_t> axes_,
                    std::vector<int64_t> sizes_,
                    bool noOverlap_,
                    const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

class DynamicZeroInplaceOp : public DynamicBinaryBaseInplaceOp {
public:
  DynamicZeroInplaceOp(const DynamicZeroOp &dynamicZeroOp);
  DynamicZeroInplaceOp(const OperatorIdentifier &_opid,
                       std::vector<int64_t> axes_,
                       std::vector<int64_t> sizes_,
                       bool noOverlap_,
                       const Op::Settings &,
                       TensorInfo updateInInfo_ = TensorInfo());
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
