// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/ir.hpp>
#include <popart/op/accumulatorupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

void AccumulatorUpdateOp::setup() {
  outInfo(getUpdatedVarOutIndex()) = inInfo(getVarToUpdateInIndex());
}

std::unique_ptr<Op>
AccumulatorUpdateOp::cloneWithNewName(const TensorId &x) const {
  return std::make_unique<AccumulatorUpdateOp>(x, settings);
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
}

AccumulatorUpdateOp::AccumulatorUpdateOp(const TensorId &varToUpdate,
                                         const Op::Settings &opSettings)
    : VarUpdateOp(Onnx::CustomOperators::AccumulatorUpdate,
                  varToUpdate,
                  opSettings) {}

} // namespace popart
