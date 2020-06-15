// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

std::unique_ptr<Op> AccumulateOp::cloneWithNewName(const TensorId &x) const {
  return std::make_unique<AccumulateOp>(x, type, factor, settings);
}

std::unique_ptr<Op> AccumulateOp::clone() const {
  return std::make_unique<AccumulateOp>(*this);
}

// T12001
std::map<InIndex, TensorId> AccumulateOp::optimizerInputs() const {
  std::map<InIndex, TensorId> m;
  if (!factor.isConst()) {
    auto index = getFactorInIndex();
    m.insert({index, inId(index)});
  }
  return m;
}

void AccumulateOp::appendOutlineAttributes(OpSerialiserBase &os) const {

  Op::appendOutlineAttributes(os);

  os.appendAttribute("type", static_cast<int>(getAccumulationType()));

  if (factor.isConst()) {
    os.appendAttribute("const factor", getFactor().val());
  }
}

AccumulateOp::AccumulateOp(const TensorId &varToUpdate,
                           AccumulationType type_,
                           OptimizerValue factor_,
                           const Op::Settings &opSettings)
    : VarUpdateWithUpdaterOp(Onnx::CustomOperators::Accumulate,
                             varToUpdate,
                             opSettings),
      type(type_), factor(factor_) {}

} // namespace popart
