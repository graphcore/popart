// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <map>
#include <memory>
#include <string>
#include <popart/op/adamvarupdate.hpp>
#include <popart/opserialiser.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
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
  if (initMwn.isConst()) {
    os.appendAttribute("const max weight norm", initMwn.val());
  }
}

AdamVarUpdateOp::AdamVarUpdateOp(OptimizerValue lr,
                                 OptimizerValue mwn,
                                 const Op::Settings &opSettings)
    : VarUpdateWithUpdaterOp(Onnx::CustomOperators::AdamVarUpdate, opSettings),
      initLr(lr), initMwn(mwn) {}

} // namespace popart
