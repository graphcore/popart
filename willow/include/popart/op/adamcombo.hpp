// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADAMCOMBOOP_HPP
#define GUARD_NEURALNET_ADAMCOMBOOP_HPP

#include <popart/adam.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/optimizer.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

// The "Combo" in the name signfies that this Op will be decomposed into
// smaller Ops :
// (1) AccumulateOp        (if gradient accumulation is enabled)
// (2) AccumulateOp        (1st momentum moving average)
// (3) AccumulateOp        (2nd momentum moving average)
// (4) AccumulatorUpdate   (if gradient accumulation is enabled)
// (5) AdamUpdater         (update term)
// (6) LambSquareOp for R1 (if Lamb is used)
// (7) LambSquareOp for R2 (if Lamb is used)
// (8) AdamUpdateOp

class AdamComboOp : public VarUpdateWithUpdaterOp {
public:
  AdamComboOp(OptimizerValue initialLr,
              OptimizerValue initialWd,
              OptimizerValue initialB1,
              OptimizerValue initialB2,
              OptimizerValue initialEps,
              OptimizerValue initialLs,
              OptimizerValue mwn,
              OptimizerValue initialGs,
              AdamMode mode_,
              WeightDecayMode decayMode_,
              bool withGradAccum_,
              OptimizerReductionType reductionType_,
              DataType accumType_,
              DataType accl1Type_,
              DataType accl2Type_,
              bool scaledOptimizerState_,
              const Op::Settings &);

  std::unique_ptr<Op> clone() const final;

  // map of size 0/1/2, containing all non-const optimizer Tensors for this Op
  std::map<InIndex, TensorId> optimizerInputs() const final;

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  // learning rate
  const OptimizerValue initLr;

  // weight decay
  const OptimizerValue initWd;

  // beta 1
  const OptimizerValue initB1;

  // beta 2
  const OptimizerValue initB2;

  // eps
  const OptimizerValue initEps;

  // loss scaling
  const OptimizerValue initLs;

  // maximum trust ratio (Lamb)
  const OptimizerValue initMwn;

  // gradient scaling
  const OptimizerValue initGs;

  // Adam mode
  const AdamMode mode;

  // Weight decay mode
  const WeightDecayMode decayMode;

  // Gradient accumulation
  const bool withGradAccum;

  const OptimizerReductionType reductionType;

  // Data type of accumulator and momentum
  DataType accumType;
  DataType accl1Type;
  DataType accl2Type;

  const bool scaledOptimizerState;

  static InIndex getLrInIndex() { return 2; }
  static InIndex getWdInIndex() { return 3; }
  static InIndex getBeta1InIndex() { return 4; }
  static InIndex getBeta2InIndex() { return 5; }
  static InIndex getEpsInIndex() { return 6; }
  static InIndex getLsInIndex() { return 7; }
  static InIndex getMwnInIndex() { return 8; }
  static InIndex getGsInIndex() { return 9; }

  std::set<InIndex> optionalInputs() const final;

  // this Op should not be present when outlining is performed
  float getSubgraphValue() const final { return -1.0f; }
};

} // namespace popart

#endif
