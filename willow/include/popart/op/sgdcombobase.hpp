// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGDVARUPDATECOMBOBASEOP_HPP
#define GUARD_NEURALNET_SGDVARUPDATECOMBOBASEOP_HPP

#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/operators.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

enum class OptimizerReductionType;

class SGDMComboBaseOp : public VarUpdateWithUpdaterOp {
public:
  SGDMComboBaseOp(const OperatorIdentifier &opid,
                  OptimizerValue initialSmm1,
                  OptimizerValue initialDpsf1,
                  OptimizerValue initialSwd1,
                  OptimizerValue initialSlr1,
                  OptimizerReductionType reductionType_,
                  const Op::Settings &);

  std::unique_ptr<Op> clone() const override = 0;

  // map of size 0/1/2, containing all non-const optimizer Tensors for this Op
  std::map<InIndex, TensorId> optimizerInputs() const override;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  // momentum
  const OptimizerValue initSmm1;

  // dampening scale factor
  const OptimizerValue initDpsf1;

  // weight decay scale factor
  const OptimizerValue initSwd1;

  // scaled learning rate
  const OptimizerValue initSlr1;

  const OptimizerReductionType reductionType;

  static InIndex getSmm1InIndex() { return 2; }
  static InIndex getDpsf1InIndex() { return 3; }
  static InIndex getSwd1InIndex() { return 4; }
  static InIndex getSlr1InIndex() { return 5; }

  std::set<InIndex> optionalInputs() const override;

  // this Op should not be present when outlining is performed
  float getSubgraphValue() const override { return -1.0f; }
};

} // namespace popart

#endif
