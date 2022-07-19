// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_DYNAMIC_DYNAMICADD_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_DYNAMIC_DYNAMICADD_HPP_

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/dynamic/dynamicbase.hpp>

#include "popart/op.hpp"

namespace popart {
class TensorInfo;
struct OperatorIdentifier;

class DynamicAddOp : public DynamicTernaryBaseOp {
public:
  DynamicAddOp(const OperatorIdentifier &_opid,
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

class DynamicAddInplaceOp : public DynamicTernaryBaseInplaceOp {
public:
  DynamicAddInplaceOp(const DynamicAddOp &dynamicAddOp);
  DynamicAddInplaceOp(const OperatorIdentifier &_opid,
                      std::vector<int64_t> axes_,
                      std::vector<int64_t> sizes_,
                      bool noOverlap_,
                      const Op::Settings &settings_,
                      TensorInfo updateInInfo_ = TensorInfo());
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_DYNAMIC_DYNAMICADD_HPP_
