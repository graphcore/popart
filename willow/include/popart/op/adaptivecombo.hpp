// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADAPTIVECOMBOOP_HPP
#define GUARD_NEURALNET_ADAPTIVECOMBOOP_HPP

#include <popart/adaptive.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/optimizer.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

// The "Combo" in the name signifies that this Op will be decomposed into
// smaller Ops.

class AdaptiveComboOp : public VarUpdateWithUpdaterOp {
public:
  AdaptiveComboOp(OptimizerValue initialLr,
                  OptimizerValue initialWd,
                  OptimizerValue initialA,
                  OptimizerValue initialM,
                  OptimizerValue initialEps,
                  OptimizerValue initialLs,
                  OptimizerValue initialGs,
                  AdaptiveMode mode_,
                  WeightDecayMode decayMode_,
                  bool withGradAccum_,
                  OptimizerReductionType reductionType_,
                  DataType accumType_,
                  DataType accl1Type_,
                  DataType accl2Type_,
                  DataType accl3Type_,
                  bool rmspropTFVariant_,
                  const Op::Settings &);

  std::unique_ptr<Op> clone() const final;

  // map of size 0/1/2, containing all non-const optimizer Tensors for this Op
  std::map<InIndex, TensorId> optimizerInputs() const final;

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  // learning rate
  const OptimizerValue initLr;

  // weight decay
  const OptimizerValue initWd;

  // alpha
  const OptimizerValue initA;

  // momentum
  const OptimizerValue initM;

  // eps
  const OptimizerValue initEps;

  // loss scaling
  const OptimizerValue initLs;

  // gradient scaling
  const OptimizerValue initGs;

  // Adaptive mode
  const AdaptiveMode mode;

  // Weight decay mode
  const WeightDecayMode decayMode;

  // Gradient accumulation
  const bool withGradAccum;

  const OptimizerReductionType reductionType;

  // Data type of accumulator and momentum
  DataType accumType;
  DataType accl1Type;
  DataType accl2Type;
  DataType accl3Type;

  // Tensorflow variant of RMSProp optimizers
  const bool rmspropTFVariant;

  static InIndex getLrInIndex() { return 2; }
  static InIndex getWdInIndex() { return 3; }
  static InIndex getAlphaInIndex() { return 4; }
  static InIndex getMomentumInIndex() { return 5; }
  static InIndex getEpsInIndex() { return 6; }
  static InIndex getLsInIndex() { return 7; }
  static InIndex getGsInIndex() { return 8; }

  std::set<InIndex> optionalInputs() const final;

  // this Op should not be present when outlining is performed
  float getSubgraphValue() const final { return -1.0f; }
};

} // namespace popart

#endif
