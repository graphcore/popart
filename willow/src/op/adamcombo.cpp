// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/adamcombo.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

AdamComboOp::AdamComboOp(const TensorId &varToUpdate,
                         OptimizerValue initialLr,
                         OptimizerValue initialWd,
                         OptimizerValue initialB1,
                         OptimizerValue initialB2,
                         OptimizerValue initialEps,
                         OptimizerValue initialLs,
                         OptimizerValue initialMwn,
                         OptimizerValue initialGs,
                         AdamMode mode_,
                         WeightDecayMode decayMode_,
                         bool withGradAccum_,
                         OptimizerReductionType reductionType_,
                         DataType accumType_,
                         DataType accl1Type_,
                         DataType accl2Type_,
                         const Op::Settings &settings_)
    : VarUpdateWithUpdaterOp(Onnx::CustomOperators::AdamCombo,
                             varToUpdate,
                             settings_),
      initLr(initialLr), initWd(initialWd), initB1(initialB1),
      initB2(initialB2), initEps(initialEps), initLs(initialLs),
      initMwn(initialMwn), initGs(initialGs), mode(mode_),
      decayMode(decayMode_), withGradAccum(withGradAccum_),
      reductionType(reductionType_), accumType(accumType_),
      accl1Type(accl1Type_), accl2Type(accl2Type_) {}

void AdamComboOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  if (initLr.isConst()) {
    os.appendAttribute("const learning rate", initLr.val());
  }

  if (initWd.isConst()) {
    os.appendAttribute("const weight decay", initWd.val());
  }

  if (initB1.isConst()) {
    os.appendAttribute("const beta1", initB1.val());
  }

  if (initB2.isConst()) {
    os.appendAttribute("const beta2", initB2.val());
  }

  if (initEps.isConst()) {
    os.appendAttribute("const eps", initEps.val());
  }

  if (initLs.isConst()) {
    os.appendAttribute("const loss scaling", initLs.val());
  }

  if (initMwn.isConst()) {
    os.appendAttribute("const max weight norm", initMwn.val());
  }

  os.appendAttribute("reduction type", static_cast<int>(reductionType));
  os.appendAttribute("adam mode", static_cast<int>(mode));
  os.appendAttribute("decay mode", static_cast<int>(decayMode));
}

std::unique_ptr<Op> AdamComboOp::cloneWithNewName(const TensorId &x) const {
  return std::make_unique<AdamComboOp>(x,
                                       initLr,
                                       initWd,
                                       initB1,
                                       initB2,
                                       initEps,
                                       initLs,
                                       initMwn,
                                       initGs,
                                       mode,
                                       decayMode,
                                       withGradAccum,
                                       reductionType,
                                       accumType,
                                       accl1Type,
                                       accl2Type,
                                       settings);
}

std::unique_ptr<Op> AdamComboOp::clone() const {
  return std::make_unique<AdamComboOp>(*this);
}

std::map<InIndex, TensorId> AdamComboOp::optimizerInputs() const {

  std::map<InIndex, TensorId> m;

  if (!initLr.isConst()) {
    auto index = getLrInIndex();
    m.insert({index, inId(index)});
  }

  if (!initWd.isConst()) {
    auto index = getWdInIndex();
    m.insert({index, inId(index)});
  }

  if (!initB1.isConst()) {
    auto index = getBeta1InIndex();
    m.insert({index, inId(index)});
  }

  if (!initB2.isConst()) {
    auto index = getBeta2InIndex();
    m.insert({index, inId(index)});
  }

  if (!initEps.isConst()) {
    auto index = getEpsInIndex();
    m.insert({index, inId(index)});
  }

  if (!initLs.isConst()) {
    auto index = getLsInIndex();
    m.insert({index, inId(index)});
  }

  if (!initMwn.isConst()) {
    auto index = getMwnInIndex();
    m.insert({index, inId(index)});
  }

  if (!initGs.isConst()) {
    auto index = getGsInIndex();
    m.insert({index, inId(index)});
  }

  return m;
}

std::set<InIndex> AdamComboOp::optionalInputs() const {
  return {getLrInIndex(),
          getWdInIndex(),
          getBeta1InIndex(),
          getBeta2InIndex(),
          getWdInIndex(),
          getEpsInIndex(),
          getLsInIndex(),
          getMwnInIndex(),
          getGsInIndex()};
}

} // namespace popart
