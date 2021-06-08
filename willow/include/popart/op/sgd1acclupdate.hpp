// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD1VARUPDATEACCLUPDATEOPOP_HPP
#define GUARD_NEURALNET_SGD1VARUPDATEACCLUPDATEOPOP_HPP

#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

/**
 * @brief Performs the part of the SGD1 velocity update equation that is
 * pre-computed for the next time step after the weight update of the current
 * time step.
 *
 * Let:
 *   `v` be the input at `getVarToUpdateInIndex()`
 *   `g` be the input at `getUpdaterInIndex()`
 * then this op performs:
 *   v <- v * smm1 + swd1 * g
 *
 * \sa SGD for how this is derived and the definitions of smm1 and swd1.
 */
class SGD1AcclUpdateOp : public VarUpdateWithUpdaterOp {

public:
  SGD1AcclUpdateOp(OptimizerValue initSmm1,
                   OptimizerValue initSwd1,
                   const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  std::map<InIndex, TensorId> optimizerInputs() const override;
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  const OptimizerValue initSmm1;
  const OptimizerValue initSwd1;
  static InIndex getSmm1InIndex() { return 2; }
  static InIndex getSwd1InIndex() { return 3; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

protected:
  SGD1AcclUpdateOp(OptimizerValue initSmm1,
                   OptimizerValue initSwd1,
                   OperatorIdentifier opid,
                   const Op::Settings &);
};

} // namespace popart

#endif
