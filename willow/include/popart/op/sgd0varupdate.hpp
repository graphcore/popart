// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD0VARUPDATE_HPP
#define GUARD_NEURALNET_SGD0VARUPDATE_HPP

#include <popart/op/varupdate.hpp>
#include <popart/optimizer.hpp>

namespace popart {

class SGD0VarUpdateOpBase : public VarUpdateWithUpdaterOp {
public:
  SGD0VarUpdateOpBase(const OperatorIdentifier &_opid,
                      OptimizerValue initialSlr0,
                      OptimizerValue initialWdsf0,
                      OptimizerReductionType reductionType_,
                      const Op::Settings &settings_);

  // If the scaled learning rate is not constant, this is the index at which it
  // will be consumed by this Op
  static InIndex getSlr0InIndex() { return 2; }

  // If the weight decay scale factor is not constant, this is the index at
  // which it will be consumed by this Op
  static InIndex getWdsf0InIndex() { return 3; }

  // map of size 0/1/2, containing all non-const optimizer Tensors for this Op
  std::map<InIndex, TensorId> optimizerInputs() const final;

  OptimizerReductionType getReductionType() { return reductionType; }

  // scaled learning rate
  const OptimizerValue initSlr0;

  // weight decay scaling factor
  const OptimizerValue initWdsf0;

  // Reduction type
  const OptimizerReductionType reductionType;

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  std::set<InIndex> optionalInputs() const final;
};

// The "0" in the name signifies that there are no persistant Tensors required
// and associated to the Variable Tensor being updated. Specifically, there is
// no gradient accumulation and no momentum (momentum factor is 0)
class SGD0VarUpdateOp : public SGD0VarUpdateOpBase {
public:
  SGD0VarUpdateOp(OptimizerValue initialSlr0,
                  OptimizerValue initialWdsf0,
                  OptimizerReductionType reductionType_,
                  const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  float getSubgraphValue() const final;
};

} // namespace popart

#endif
