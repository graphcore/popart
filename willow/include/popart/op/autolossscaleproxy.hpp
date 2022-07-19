// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef POPART_WILLOW_INCLUDE_POPART_OP_AUTOLOSSSCALEPROXY_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_AUTOLOSSSCALEPROXY_HPP_

#include <map>
#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {
struct OperatorIdentifier;

// This op labels the user provided forward tensors for automatic loss scaling.
// This no-op helps to find the associated gradient tensors.

class AutoLossScaleProxyOp : public ElementWiseUnaryOp {
public:
  AutoLossScaleProxyOp(const OperatorIdentifier &_opid,
                       const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

// This grad op labels corresponding tracked gradient tensors
// for automatic loss scaling.
class AutoLossScaleProxyGradOp : public AutoLossScaleProxyOp {
public:
  AutoLossScaleProxyGradOp(const AutoLossScaleProxyOp &fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_AUTOLOSSSCALEPROXY_HPP_
