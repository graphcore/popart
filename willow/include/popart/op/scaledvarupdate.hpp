// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SCALEDVARUPDATE_HPP
#define GUARD_NEURALNET_SCALEDVARUPDATE_HPP

#include <popart/op/varupdate.hpp>

namespace popart {

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

#endif
