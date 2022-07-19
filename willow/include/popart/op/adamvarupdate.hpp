// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_ADAMVARUPDATE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_ADAMVARUPDATE_HPP_

#include <map>
#include <memory>
#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
class OpSerialiserBase;

class AdamVarUpdateOp : public VarUpdateWithUpdaterOp {
public:
  AdamVarUpdateOp(OptimizerValue initLr,
                  OptimizerValue mwn,
                  const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  std::map<InIndex, TensorId> optimizerInputs() const final;
  void appendOutlineAttributes(OpSerialiserBase &) const final;

  const OptimizerValue initLr;
  const OptimizerValue initMwn;

  static InIndex getLambR1SqInIndex() { return 2; }
  static InIndex getLambR2SqInIndex() { return 3; }
  static InIndex getLrInIndex() { return 4; }
  static InIndex getMwnInIndex() { return 5; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_ADAMVARUPDATE_HPP_
