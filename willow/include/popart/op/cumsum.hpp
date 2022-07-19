// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_CUMSUM_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_CUMSUM_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <popart/op.hpp>

#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

// Performs cumulative sum of the input elements along the given axis.
// By default, it will do the sum inclusively meaning the first element
// is copied as is. Through an exclusive attribute, this behavior can change
// to exclude the first element. It can also perform summation in the
// opposite direction of the axis. For that, set reverse attribute to 1.

class CumSumOp : public Op {
public:
  CumSumOp(const OperatorIdentifier &_opid,
           bool exclusive_,
           bool reverse_,
           const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() override;
  void setup() final;

  bool getExclusive() const;
  bool getReverse() const;
  int64_t getAxis() const;

  static InIndex xInIndex() { return 0; }
  static InIndex axisInIndex() { return 1; }
  static OutIndex outIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  int64_t axis         = 0;
  const bool exclusive = false;
  const bool reverse   = false;
};

class CumSumGradOp : public Op {
public:
  CumSumGradOp(const CumSumOp &op, bool exclusive, bool reverse, int64_t axis);

  std::unique_ptr<Op> clone() const override;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  bool getExclusive() const;
  bool getReverse() const;
  int64_t getAxis() const;

  static InIndex outGradXInIndex() { return 0; }
  static InIndex fwdXInIndex() { return 1; }
  static OutIndex outIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  const bool exclusive;
  const bool reverse;
  int64_t axis;
  const TensorInfo fwdOpXInInfo;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_CUMSUM_HPP_
