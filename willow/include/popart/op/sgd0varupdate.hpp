// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SGD0VARUPDATE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SGD0VARUPDATE_HPP_

#include <map>
#include <memory>
#include <set>
#include <popart/op/varupdate.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class SGD0VarUpdateOpBase : public VarUpdateWithUpdaterOp {
public:
  SGD0VarUpdateOpBase(const OperatorIdentifier &_opid,
                      OptimizerValue initialSlr0,
                      OptimizerValue initialWdsf0,
                      const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override = 0;

  // If the scaled learning rate is not constant, this is the index at which it
  // will be consumed by this Op
  static InIndex getSlr0InIndex() { return 2; }

  // If the weight decay scale factor is not constant, this is the index at
  // which it will be consumed by this Op
  static InIndex getWdsf0InIndex() { return 3; }

  // map of size 0/1/2, containing all non-const optimizer Tensors for this Op
  std::map<InIndex, TensorId> optimizerInputs() const final;

  // scaled learning rate
  const OptimizerValue initSlr0;

  // weight decay scaling factor
  const OptimizerValue initWdsf0;

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  std::set<InIndex> optionalInputs() const final;
};

class SGD0VarUpdateOp : public SGD0VarUpdateOpBase {
public:
  SGD0VarUpdateOp(OptimizerValue initialSlr0,
                  OptimizerValue initialWdsf0,
                  const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  float getSubgraphValue() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SGD0VARUPDATE_HPP_
