// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SPLINEWEIGHTING_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SPLINEWEIGHTING_HPP_

#include <memory>
#include <popart/op.hpp>

namespace popart {

class SplineWeightingOp : public Op {
public:
  SplineWeightingOp(const OperatorIdentifier &opid,
                    const Op::Settings &settings);

  static constexpr InIndex inputIndex() noexcept { return 0; }
  static constexpr InIndex weightIndex() noexcept { return 1; }
  static constexpr InIndex basisIndex() noexcept { return 2; }
  static constexpr InIndex weightIndexIndex() noexcept { return 3; }
  static constexpr OutIndex outputIndex() noexcept { return 0; }

  void setup() override;
  std::unique_ptr<Op> clone() const override;
  float getSubgraphValue() const override;
  void appendOutlineAttributes(OpSerialiserBase &) const override;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SPLINEWEIGHTING_HPP_
