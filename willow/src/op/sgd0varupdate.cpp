#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

void SGD0VarUpdateOpBase::appendAttributes(OpSerialiserBase &os) const {

  Op::appendAttributes(os);

  if (initSlr0.isConst()) {
    os.appendAttribute("const scaled learning rate", initSlr0.val());
  }

  if (initWdsf0.isConst()) {
    os.appendAttribute("const weight decay scale factor", initWdsf0.val());
  }
}

SGD0VarUpdateOpBase::SGD0VarUpdateOpBase(const OperatorIdentifier &opid,
                                         const TensorId &varId_,
                                         OptimizerValue initialSlr0,
                                         OptimizerValue initialWdsf0,
                                         const Op::Settings &settings_)
    : VarUpdateWithUpdaterOp(opid, varId_, settings_), initSlr0(initialSlr0),
      initWdsf0(initialWdsf0) {}

std::map<InIndex, TensorId> SGD0VarUpdateOpBase::optimizerInputs() const {
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
    : SGD0VarUpdateOpBase(Onnx::CustomOperators::SGD0VarUpdate,
                          varId_,
                          slr0,
                          wdsf0,
                          settings_) {}

} // namespace popart
