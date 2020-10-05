// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CUMSUM_HPP
#define GUARD_NEURALNET_CUMSUM_HPP

#include <popart/op.hpp>

namespace popart {

class CumSumOp : public Op {
public:
  CumSumOp(const OperatorIdentifier &_opid,
           int64_t exclusive_,
           int64_t reverse_,
           const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  int64_t getExclusive() const;
  int64_t getReverse() const;
  int64_t getAxis() const;

  static InIndex xInIndex() { return 0; }
  static InIndex axisInIndex() { return 1; }
  static OutIndex outIndex() { return 0; }

  void setAxis(int64_t x) { axis = x; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  int64_t axis      = 0;
  int64_t exclusive = 0;
  int64_t reverse   = 0;
};

} // namespace popart

#endif
