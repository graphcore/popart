// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <map>
#include <memory>
#include <popart/op/accumulatorscale.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {

std::unique_ptr<Op> AccumulatorScaleOp::clone() const {
  return std::make_unique<AccumulatorScaleOp>(*this);
}

std::map<InIndex, TensorId> AccumulatorScaleOp::optimizerInputs() const {
  std::map<InIndex, TensorId> m;
  return m;
}

void AccumulatorScaleOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);

  if (factor.isConst()) {
    os.appendAttribute("const factor", factor.val());
  }
}

AccumulatorScaleOp::AccumulatorScaleOp(const OptimizerValue factor_,
                                       const Op::Settings &opSettings)
    : VarUpdateOp(Onnx::CustomOperators::AccumulatorScale, opSettings),
      factor(factor_) {}

view::Regions AccumulatorScaleOp::modifies(InIndex index) const {
  if (factor.isConst()) {
    if (factor.val() == 0.0f) {
      return {view::Region::getFull(inShape(index), view::AccessType::Write)};
    }
  }
  return {view::Region::getFull(inShape(index), view::AccessType::ReadWrite)};
}

} // namespace popart
