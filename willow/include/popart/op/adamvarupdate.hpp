// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADAMVARUPDATE_HPP
#define GUARD_NEURALNET_ADAMVARUPDATE_HPP

#include <popart/op/varupdate.hpp>

namespace popart {

class AdamVarUpdateOp : public VarUpdateWithUpdaterOp {

public:
  AdamVarUpdateOp(const TensorId &varToUpdate,
                  OptimizerValue initLr,
                  OptimizerValue mwn,
                  const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  std::unique_ptr<Op> cloneWithNewName(const TensorId &newName) const final;
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

#endif
