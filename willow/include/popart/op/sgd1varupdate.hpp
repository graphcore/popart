// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD1VARUPDATE_HPP
#define GUARD_NEURALNET_SGD1VARUPDATE_HPP

#include <popart/op/varupdate.hpp>

namespace popart {

class SGD1VarUpdateOp : public VarUpdateWithUpdaterOp {

public:
  SGD1VarUpdateOp(OptimizerValue initSlr1, const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  std::map<InIndex, TensorId> optimizerInputs() const final;
  void appendOutlineAttributes(OpSerialiserBase &) const final;

  const OptimizerValue initSlr1;
  static InIndex getSlr1InIndex() { return 2; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif
