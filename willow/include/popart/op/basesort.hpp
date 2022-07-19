// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_BASESORT_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_BASESORT_HPP_

#include <cstdint>
#include <memory>
#include <popart/op.hpp>

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class BaseSortOp : public Op {
public:
  BaseSortOp(const OperatorIdentifier &_opid,
             int64_t axis,
             const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;

  int64_t getAxis() const;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  static int getInIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

protected:
  // confirm that the axis is within the input tensor's rank
  void validateAxis() const;

private:
  const int64_t axis;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_BASESORT_HPP_
