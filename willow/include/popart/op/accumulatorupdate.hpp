// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ACCUMULATORUPDATEOP_HPP
#define GUARD_NEURALNET_ACCUMULATORUPDATEOP_HPP

#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

// Update accumulator by dividing it by the replication factor in case of
// multi-replica gradient accumulation
class AccumulatorUpdateOp : public VarUpdateOp {

public:
  AccumulatorUpdateOp(const TensorId &varToUpdate, const Op::Settings &);
  void setup() final;
  std::unique_ptr<Op> clone() const final;
  std::unique_ptr<Op> cloneWithNewName(const TensorId &newName) const final;
  std::map<InIndex, TensorId> optimizerInputs() const final;
  void appendOutlineAttributes(OpSerialiserBase &) const final;
  static InIndex getFactorInIndex() { return 2; }
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif
