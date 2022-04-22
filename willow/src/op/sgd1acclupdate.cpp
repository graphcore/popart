// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <popart/op/sgd1acclupdate.hpp>
#include <popart/opserialiser.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {

std::unique_ptr<Op> SGD1AcclUpdateOp::clone() const {
  return std::make_unique<SGD1AcclUpdateOp>(*this);
}

std::map<InIndex, TensorId> SGD1AcclUpdateOp::optimizerInputs() const {
  std::map<InIndex, TensorId> m;
  if (!initSmm1.isConst()) {
    auto index = getSmm1InIndex();
    m.insert({index, inId(index)});
  }
  if (!initSwd1.isConst()) {
    auto index = getSwd1InIndex();
    m.insert({index, inId(index)});
  }
  return m;
}

void SGD1AcclUpdateOp::appendOutlineAttributes(OpSerialiserBase &os) const {

  Op::appendOutlineAttributes(os);

  if (initSmm1.isConst()) {
    os.appendAttribute("const momentum", initSmm1.val());
  }

  if (initSwd1.isConst()) {
    os.appendAttribute("const weight decay scale factor", initSwd1.val());
  }
}

SGD1AcclUpdateOp::SGD1AcclUpdateOp(OptimizerValue smm1,
                                   OptimizerValue swd1,
                                   const Op::Settings &opSettings)
    : VarUpdateWithUpdaterOp(Onnx::CustomOperators::SGD1AcclUpdate, opSettings),
      initSmm1(smm1), initSwd1(swd1) {}

SGD1AcclUpdateOp::SGD1AcclUpdateOp(OptimizerValue smm1,
                                   OptimizerValue swd1,
                                   OperatorIdentifier opid,
                                   const Op::Settings &opSettings)
    : VarUpdateWithUpdaterOp(std::move(opid), opSettings),
      initSmm1(std::move(smm1)), initSwd1(std::move(swd1)) {}

} // namespace popart
