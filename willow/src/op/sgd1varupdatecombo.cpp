#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/sgd1varupdatecombo.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

SGD1VarUpdateComboOp::SGD1VarUpdateComboOp(const TensorId &varId_,
                                           OptimizerValue initialMm1,
                                           OptimizerValue initialDpsf1,
                                           OptimizerValue initialWdsf1,
                                           OptimizerValue initialSlr1,
                                           const Op::Settings &settings_)
    : VarUpdateOp(Onnx::CustomOperators::SGD1VarUpdateCombo, varId_, settings_),
      initMm1(initialMm1), initDpsf1(initialDpsf1), initWdsf1(initialWdsf1),
      initSlr1(initialSlr1) {}

void SGD1VarUpdateComboOp::appendAttributes(OpSerialiserBase &os) const {

  Op::appendAttributes(os);

  if (initMm1.isConst()) {
    os.appendAttribute("const momentum", initMm1.val());
  }

  if (initDpsf1.isConst()) {
    os.appendAttribute("const dampening scale factor", initDpsf1.val());
  }

  if (initWdsf1.isConst()) {
    os.appendAttribute("const weight decay scale factor", initWdsf1.val());
  }

  if (initSlr1.isConst()) {
    os.appendAttribute("const scaled learning rate", initSlr1.val());
  }
}

std::unique_ptr<Op>
SGD1VarUpdateComboOp::cloneWithNewName(const TensorId &x) const {
  return std::make_unique<SGD1VarUpdateComboOp>(
      x, initMm1, initDpsf1, initWdsf1, initSlr1, settings);
}

std::unique_ptr<Op> SGD1VarUpdateComboOp::clone() const {
  return std::make_unique<SGD1VarUpdateComboOp>(*this);
}

std::map<InIndex, TensorId> SGD1VarUpdateComboOp::optimizerInputs() const {

  std::map<InIndex, TensorId> m;

  if (!initSlr1.isConst()) {
    auto index = getSlr1InIndex();
    m.insert({index, inId(index)});
  }

  if (!initWdsf1.isConst()) {
    auto index = getWdsf1InIndex();
    m.insert({index, inId(index)});
  }

  if (!initMm1.isConst()) {
    auto index = getMm1InIndex();
    m.insert({index, inId(index)});
  }

  if (!initDpsf1.isConst()) {
    auto index = getDpsf1InIndex();
    m.insert({index, inId(index)});
  }

  return m;
}

} // namespace popart
