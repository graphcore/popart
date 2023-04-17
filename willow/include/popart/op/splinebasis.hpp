// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SPLINEBASIS_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SPLINEBASIS_HPP_

#include <memory>
#include <popart/op.hpp>

namespace popart {

class SplineBasisOp : public Op {
public:
  SplineBasisOp(const OperatorIdentifier &opid,
                int degree,
                const Op::Settings &settings);

  static constexpr InIndex pseudoIndex() noexcept { return 0; }
  static constexpr InIndex kernelSizeIndex() noexcept { return 1; }
  static constexpr InIndex isOpenSplineIndex() noexcept { return 2; }
  static constexpr OutIndex outBasisIndex() noexcept { return 0; }
  static constexpr OutIndex outWeightIndexIndex() noexcept { return 1; }

  void setup() override;
  std::unique_ptr<Op> clone() const override;
  float getSubgraphValue() const override;
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  unsigned getDegree() const noexcept;

private:
  int degree_;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SPLINEBASIS_HPP_
