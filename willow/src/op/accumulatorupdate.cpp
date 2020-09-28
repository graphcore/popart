// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/ir.hpp>
#include <popart/op/accumulatorupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

std::unique_ptr<Op>
AccumulatorUpdateOp::cloneWithNewName(const TensorId &x) const {
  return std::make_unique<AccumulatorUpdateOp>(x, factor, settings);
}

std::unique_ptr<Op> AccumulatorUpdateOp::clone() const {
  return std::make_unique<AccumulatorUpdateOp>(*this);
}

std::map<InIndex, TensorId> AccumulatorUpdateOp::optimizerInputs() const {
  std::map<InIndex, TensorId> m;
  return m;
}

void AccumulatorUpdateOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);

  if (factor.isConst()) {
    os.appendAttribute("const factor", factor.val());
  }
}

AccumulatorUpdateOp::AccumulatorUpdateOp(const TensorId &varToUpdate,
                                         const OptimizerValue factor_,
                                         const Op::Settings &opSettings)
    : VarUpdateWithoutUpdaterOp(Onnx::CustomOperators::AccumulatorUpdate,
                                varToUpdate,
                                opSettings),
      factor(factor_) {}

view::Regions AccumulatorUpdateOp::modifies(InIndex index) const {
  if (factor.isConst()) {
    if (factor.val() == 0.0f) {
      return {view::Region::getFull(inShape(index), view::AccessType::Write)};
    }
  }
  return {view::Region::getFull(inShape(index), view::AccessType::ReadWrite)};
}

} // namespace popart
