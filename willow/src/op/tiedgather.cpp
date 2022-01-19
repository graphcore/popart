// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op/tiedgather.hpp>

#include <popart/error.hpp>
#include <popart/operators.hpp>

#include <memory>
#include <vector>

namespace popart {

TiedGatherOp::TiedGatherOp(
    const int64_t axis_,
    const Op::Settings &settings_,
    const nonstd::optional<float> available_memory_proportion_,
    bool zeroOutOfRangeIndices_)
    : GatherOp(Onnx::CustomOperators::TiedGather,
               axis_,
               settings_,
               available_memory_proportion_,
               zeroOutOfRangeIndices_) {}

std::unique_ptr<Op> TiedGatherOp::clone() const {
  return std::make_unique<TiedGatherOp>(*this);
}

std::vector<std::unique_ptr<Op>> TiedGatherOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.push_back(std::make_unique<TiedGatherGradOp>(this, getAxis()));

  // In TiedGatherPattern, we insert a Detach after the TiedGatherOp (because
  // we will manually grow the backward pass in TiedGatherAccumulatePattern),
  // meaning there will be nothing after the TiedGatherGradOp in the graph, so
  // we must ensure it is not pruned before TiedGatherAccumulatePattern is run.
  result[0]->pruneable = false;

  return result;
}

TiedGatherGradOp::TiedGatherGradOp(const TiedGatherOp *fwdOp_,
                                   const int64_t axis_)
    : GatherGradOp(*fwdOp_, axis_), fwdOp(fwdOp_) {
  if (!fwdOp) {
    throw internal_error("TiedGatherGradOp ctor passed null fwdOp.");
  }
}

std::unique_ptr<Op> TiedGatherGradOp::clone() const {
  return std::make_unique<TiedGatherGradOp>(*this);
}

} // namespace popart
