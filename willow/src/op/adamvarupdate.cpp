// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/adamvarupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

std::unique_ptr<Op> AdamVarUpdateOp::cloneWithNewName(const TensorId &x) const {
  return std::make_unique<AdamVarUpdateOp>(x, initLr, settings);
}

std::unique_ptr<Op> AdamVarUpdateOp::clone() const {
  return std::make_unique<AdamVarUpdateOp>(*this);
}

std::map<InIndex, TensorId> AdamVarUpdateOp::optimizerInputs() const {
  std::map<InIndex, TensorId> m;
  if (!initLr.isConst()) {
    auto index = getLrInIndex();
    m.insert({index, inId(index)});
  }
  return m;
}

void AdamVarUpdateOp::appendOutlineAttributes(OpSerialiserBase &os) const {

  Op::appendOutlineAttributes(os);

  if (initLr.isConst()) {
    os.appendAttribute("const learning rate", initLr.val());
  }
}

AdamVarUpdateOp::AdamVarUpdateOp(const TensorId &varToUpdate,
                                 OptimizerValue lr,
                                 const Op::Settings &opSettings)
    : VarUpdateWithUpdaterOp(Onnx::CustomOperators::AdamVarUpdate,
                             varToUpdate,
                             opSettings),
      initLr(lr) {}

} // namespace popart
