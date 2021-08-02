// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/sgd0combo.hpp>
#include <popart/opserialiser.hpp>
#include <popart/optimizer.hpp>

namespace popart {

SGD0ComboOp::SGD0ComboOp(OptimizerValue initialSlr0,
                         OptimizerValue initialWdsf0,
                         bool withGradAccum_,
                         OptimizerReductionType reductionType_,
                         DataType accumType_,
                         const Op::Settings &settings_)
    : VarUpdateWithUpdaterOp(Onnx::CustomOperators::SGD0Combo, settings_),
      initSlr0(std::move(initialSlr0)), initWdsf0(std::move(initialWdsf0)),
      withGradAccum(withGradAccum_), reductionType(reductionType_),
      accumType(accumType_) {
  if (!withGradAccum && reductionType == OptimizerReductionType::AccumReduce) {
    throw error("SGD0 cannot perform OptimizerReductionType::AccumReduce when "
                "gradient accumulation is not enabled, as there will be no "
                "accum tensor to reduce.");
  }

  const auto validateIsFloatingPoint = [](const DataType dt,
                                          const std::string name) {
    if (!((dt == DataType::FLOAT) || (dt == DataType::FLOAT16))) {
      throw error("SGD0 only supports FLOAT or FLOAT16 for DataType of {}. You "
                  "passed {}.",
                  name,
                  dt);
    }
  };
  validateIsFloatingPoint(accumType, "accum");
}

void SGD0ComboOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  if (initWdsf0.isConst()) {
    os.appendAttribute("const weight decay scale factor", initWdsf0.val());
  }

  if (initSlr0.isConst()) {
    os.appendAttribute("const scaled learning rate", initSlr0.val());
  }

  os.appendAttribute("reduction type", static_cast<int>(reductionType));
}

std::map<InIndex, TensorId> SGD0ComboOp::optimizerInputs() const {

  std::map<InIndex, TensorId> m;

  if (!initSlr0.isConst()) {
    auto index = getSlr0InIndex();
    m.insert({index, inId(index)});
  }

  if (!initWdsf0.isConst()) {
    auto index = getWdsf0InIndex();
    m.insert({index, inId(index)});
  }

  return m;
}

std::set<InIndex> SGD0ComboOp::optionalInputs() const {
  return {getSlr0InIndex(), getWdsf0InIndex()};
}

std::unique_ptr<Op> SGD0ComboOp::clone() const {
  return std::make_unique<SGD0ComboOp>(*this);
}

} // namespace popart
