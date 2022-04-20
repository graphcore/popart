// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <map>
#include <memory>
#include <string>
#include <popart/op/scaledvarupdate.hpp>
#include <popart/opserialiser.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/tensordebuginfo.hpp"

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
  os.appendAttribute("learning rate in updater", static_cast<int>(lrInUpdater));
}

ScaledVarUpdateOp::ScaledVarUpdateOp(OptimizerValue lr,
                                     OptimizerValue wd,
                                     bool lrInUpdater,
                                     const Op::Settings &opSettings)
    : VarUpdateWithUpdaterOp(Onnx::CustomOperators::ScaledVarUpdate,
                             opSettings),
      initLr(lr), initWd(wd), lrInUpdater(lrInUpdater) {}

} // namespace popart
