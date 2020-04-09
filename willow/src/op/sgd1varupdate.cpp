// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

std::unique_ptr<Op> SGD1VarUpdateOp::cloneWithNewName(const TensorId &x) const {
  return std::make_unique<SGD1VarUpdateOp>(x, initSlr1, settings);
}

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

SGD1VarUpdateOp::SGD1VarUpdateOp(const TensorId &varToUpdate,
                                 OptimizerValue slr1,
                                 const Op::Settings &opSettings)
    : VarUpdateWithUpdaterOp(Onnx::CustomOperators::SGD1VarUpdate,
                             varToUpdate,
                             opSettings),
      initSlr1(slr1) {}

namespace {} // namespace

} // namespace popart
