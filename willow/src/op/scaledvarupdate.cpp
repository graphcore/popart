// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/scaledvarupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {
std::unique_ptr<Op> ScaledVarUpdateOp::clone() const {
  return std::make_unique<ScaledVarUpdateOp>(*this);
}

std::map<InIndex, TensorId> ScaledVarUpdateOp::optimizerInputs() const {
  std::map<InIndex, TensorId> m;
  if (!initLr.isConst()) {
    auto index = getLrInIndex();
    m.insert({index, inId(index)});
  }
  if (!initWd.isConst()) {
    auto index = getWdInIndex();
    m.insert({index, inId(index)});
  }
  return m;
}

void ScaledVarUpdateOp::appendOutlineAttributes(OpSerialiserBase &os) const {

  Op::appendOutlineAttributes(os);

  if (initLr.isConst()) {
    os.appendAttribute("const learning rate", initLr.val());
  }
  if (initWd.isConst()) {
    os.appendAttribute("const weight decay", initWd.val());
  }
}

ScaledVarUpdateOp::ScaledVarUpdateOp(OptimizerValue lr,
                                     OptimizerValue wd,
                                     const Op::Settings &opSettings)
    : VarUpdateWithUpdaterOp(Onnx::CustomOperators::ScaledVarUpdate,
                             opSettings),
      initLr(lr), initWd(wd) {}

} // namespace popart
