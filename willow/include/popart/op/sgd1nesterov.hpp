// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD1NESTEROV_HPP
#define GUARD_NEURALNET_SGD1NESTEROV_HPP

#include <memory>
#include <tuple>
#include <vector>
#include <poprithms/memory/inplace/proposal.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {

class OpSerialiserBase;
struct OperatorIdentifier;

/**
 * @brief Performs the part of the SGD nesterov momentum gradient update
 * equation.
 *
 * Let:
 *   `g` be the input at `getGradInIndex()` - Gradient
 *   `w` be the input at `getWeightInIndex()` - Weight
 *   `v` be the input at `getVelocityInIndex()` - Velocity
 *
 *   `ils` be the input at `getInverseLossScaleInIndex()` - Inverse loss scale
 *   `wd` be the input at `getWdInIndex()` - Weight Decay
 *   `ngsf` be the input at `getWdInIndex()` - Nesterov gradient scale factor
 *   `mm` be the input at `getMmInIndex()` - Momentum
 *
 *   `g_out` be the output at `getOutIndex()` - Updated gradient
 *
 * then this op performs:
 *   g_out <- ngsf * (ils * g + wd * w) + mm * v
 */
class SGD1NesterovOp : public Op {
public:
  SGD1NesterovOp(const OperatorIdentifier &_opid,
                 float initInverseLossScale,
                 float initWd,
                 float initNgsf,
                 float initMm,
                 const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  void setup() override;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getWeightInIndex() { return 1; }
  static InIndex getVelocityInIndex() { return 2; }
  static InIndex getInverseLossScaleInIndex() { return 3; }
  static InIndex getWdInIndex() { return 4; }
  static InIndex getNgsfInIndex() { return 5; }
  static InIndex getMmInIndex() { return 6; }
  static OutIndex getOutIndex() { return 0; }

  float getInverseLossScale() const { return initInverseLossScale; }
  float getWd() const { return initWd; }
  float getNgsf() const { return initNgsf; }
  float getMm() const { return initMm; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  virtual bool isOptimizerOp() const override { return true; }

  float getSubgraphValue() const override { return getLowSubgraphValue(); }

protected:
  float initInverseLossScale;
  float initWd;
  float initNgsf;
  float initMm;
};

} // namespace popart

#endif
