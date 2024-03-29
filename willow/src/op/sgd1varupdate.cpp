// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <map>
#include <memory>
#include <string>
#include <popart/op/sgd1varupdate.hpp>
#include <popart/opserialiser.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {

std::unique_ptr<Op> SGD1VarUpdateOp::clone() const {
  return std::make_unique<SGD1VarUpdateOp>(*this);
}

std::map<InIndex, TensorId> SGD1VarUpdateOp::optimizerInputs() const {
  std::map<InIndex, TensorId> m;
  if (!initSlr1.isConst()) {
    auto index = getSlr1InIndex();
    m.insert({index, inId(index)});
  }
  return m;
}

void SGD1VarUpdateOp::appendOutlineAttributes(OpSerialiserBase &os) const {

  Op::appendOutlineAttributes(os);

  if (initSlr1.isConst()) {
    os.appendAttribute("const scaled learning rate", initSlr1.val());
  }
}

SGD1VarUpdateOp::SGD1VarUpdateOp(OptimizerValue slr1,
                                 const Op::Settings &opSettings)
    : VarUpdateWithUpdaterOp(Onnx::CustomOperators::SGD1VarUpdate, opSettings),
      initSlr1(slr1) {}

namespace {} // namespace

} // namespace popart
