#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

void SGD0VarUpdateOp::appendAttributes(OpSerialiserBase &os) const {

  Op::appendAttributes(os);

  if (initSlr0.isConst()) {
    os.appendAttribute("const scaled learning rate", initSlr0.val());
  }

  if (initWdsf0.isConst()) {
    os.appendAttribute("const weight decay scale factor", initWdsf0.val());
  }
}

std::unique_ptr<Op> SGD0VarUpdateOp::cloneWithNewName(const TensorId &x) const {
  return std::make_unique<SGD0VarUpdateOp>(x, initSlr0, initWdsf0, settings);
}

std::unique_ptr<Op> SGD0VarUpdateOp::clone() const {
  return std::make_unique<SGD0VarUpdateOp>(*this);
}

SGD0VarUpdateOp::SGD0VarUpdateOp(const TensorId &varId_,
                                 OptimizerValue slr0,
                                 OptimizerValue wdsf0,
                                 const Op::Settings &settings_)
    : VarUpdateOp(Onnx::CustomOperators::SGD0VarUpdate, varId_, settings_),
      initSlr0(slr0), initWdsf0(wdsf0) {}

std::map<InIndex, TensorId> SGD0VarUpdateOp::optimizerInputs() const {
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

} // namespace popart
