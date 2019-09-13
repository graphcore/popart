#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {
VarUpdateOp::VarUpdateOp(const OperatorIdentifier &_opid,
                         TensorId varId_,
                         const Op::Settings &settings_)
    : Op(_opid, settings_), varId(varId_), varGradId(getGradId(varId)) {
  // very high priority, so that performed as early as possible
  priority = std::numeric_limits<double>::max();
}

void VarUpdateOp::setup() {
  outInfo(getUpdatedVarOutIndex()) = inInfo(getVarToUpdateInIndex());
}

view::Region VarUpdateOp::aliases(InIndex index) const {
  if (index == getVarToUpdateInIndex()) {
    return view::Region::getFull(inShape(index));
  } else {
    return view::Region::getEmpty(inRank(index));
  }
}

view::Region VarUpdateOp::modifies(InIndex index) const {
  return aliases(index);
}

void SGDVarUpdateOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  if (initScaledLearningRate.isConst()) {
    os.appendAttribute("const scaled learning rate",
                       initScaledLearningRate.val());
  }

  if (initWeightDecayScaleFactor.isConst()) {
    os.appendAttribute("const weight decay scale factor",
                       initWeightDecayScaleFactor.val());
  }
}

float VarUpdateOp::getSubgraphValue() const {
  // If we have replicated graphs then outline VaruUdates, if possible
  // The motivation for this is the (code) cost of inter-IPU copies, hmm
  if (getIr().getSessionOptions().enableReplicatedGraphs) {
    return getHighSubgraphValue();
  } else {
    return getLowSubgraphValue();
  }
}

std::unique_ptr<Op> SGDVarUpdateOp::cloneWithNewName(const TensorId &x) const {
  return std::make_unique<SGDVarUpdateOp>(
      x, initScaledLearningRate, initWeightDecayScaleFactor, settings);
}

std::unique_ptr<Op> SGDVarUpdateOp::clone() const {
  return std::make_unique<SGDVarUpdateOp>(*this);
}

std::map<InIndex, TensorId> SGDVarUpdateOp::optimizerInputs() const {
  std::map<InIndex, TensorId> m;
  if (!initScaledLearningRate.isConst()) {
    auto index = getScaledLearningRateInIndex();
    m.insert({index, inId(index)});
  }
  if (!initWeightDecayScaleFactor.isConst()) {
    auto index = getWeightDecayScaleFactorInIndex();
    m.insert({index, inId(index)});
  }
  return m;
}

SGDVarUpdateOp::SGDVarUpdateOp(const TensorId &varId_,
                               OptimizerValue slr,
                               OptimizerValue wdsf,
                               const Op::Settings &settings_)
    : VarUpdateOp(Onnx::CustomOperators::SgdVarUpdate, varId_, settings_),
      initScaledLearningRate(slr), initWeightDecayScaleFactor(wdsf) {}

CopyVarUpdateOp::CopyVarUpdateOp(TensorId varId_, const Op::Settings &settings_)
    : VarUpdateOp(Onnx::CustomOperators::CopyVarUpdate, varId_, settings_) {}

std::unique_ptr<Op> CopyVarUpdateOp::clone() const {
  return std::make_unique<CopyVarUpdateOp>(*this);
}

namespace {} // namespace

} // namespace popart
