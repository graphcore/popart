// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SGDCOMBOBASE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SGDCOMBOBASE_HPP_

#include <map>
#include <memory>
#include <set>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

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

  SGDMComboBaseOp(const OperatorIdentifier &opid,
                  OptimizerValue initialSmm1,
                  OptimizerValue initialDpsf1,
                  OptimizerValue initialSwd1,
                  OptimizerValue initialSlr1,
                  OptimizerValue initialMm,
                  OptimizerValue initialWd,
                  OptimizerValue initialNgsf,
                  OptimizerValue initialNdsf,
                  OptimizerReductionType reductionType_,
                  const Op::Settings &);

  std::unique_ptr<Op> clone() const override = 0;

  // map of size 0/1/2, containing all non-const optimizer Tensors for this Op
  std::map<InIndex, TensorId> optimizerInputs() const override;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  // scaled momentum
  const OptimizerValue initSmm1;

  // dampening scale factor
  const OptimizerValue initDpsf1;

  // weight decay scale factor
  const OptimizerValue initSwd1;

  // scaled learning rate
  const OptimizerValue initSlr1;

  // momentum
  OptimizerValue initMm;

  // weight decay
  OptimizerValue initWd;

  // nesterov gradient scale factor
  OptimizerValue initNgsf;

  // nesterov dampening scale factor
  OptimizerValue initNdsf;

  const OptimizerReductionType reductionType;

  // Option to enable Nesterov momentum
  bool nesterov;

  static InIndex getSmm1InIndex() { return 2; }
  static InIndex getDpsf1InIndex() { return 3; }
  static InIndex getSwd1InIndex() { return 4; }
  static InIndex getSlr1InIndex() { return 5; }
  static InIndex getMmInIndex() { return 6; }
  static InIndex getWdInIndex() { return 7; }
  static InIndex getNgsfInIndex() { return 8; }
  static InIndex getNdsfInIndex() { return 9; }

  std::set<InIndex> optionalInputs() const override;

  // this Op should not be present when outlining is performed
  float getSubgraphValue() const override { return -1.0f; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SGDCOMBOBASE_HPP_
