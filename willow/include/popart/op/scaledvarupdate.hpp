// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SCALEDVARUPDATE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SCALEDVARUPDATE_HPP_

#include <map>
#include <memory>
#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
class OpSerialiserBase;

class ScaledVarUpdateOp : public VarUpdateWithUpdaterOp {
public:
  ScaledVarUpdateOp(OptimizerValue initLr,
                    OptimizerValue initWd,
                    bool lrInUpdater,
                    const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  std::map<InIndex, TensorId> optimizerInputs() const final;
  void appendOutlineAttributes(OpSerialiserBase &) const final;

  const OptimizerValue initLr;
  const OptimizerValue initWd;
  const bool lrInUpdater;

  static InIndex getLrInIndex() { return 2; }
  static InIndex getWdInIndex() { return 3; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SCALEDVARUPDATE_HPP_
