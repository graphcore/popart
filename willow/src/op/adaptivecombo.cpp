// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/adaptivecombo.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

AdaptiveComboOp::AdaptiveComboOp(OptimizerValue initialLr,
                                 OptimizerValue initialWd,
                                 OptimizerValue initialA,
                                 OptimizerValue initialM,
                                 OptimizerValue initialEps,
                                 OptimizerValue initialLs,
                                 OptimizerValue initialGs,
                                 AdaptiveMode mode_,
                                 WeightDecayMode decayMode_,
                                 bool withGradAccum_,
                                 OptimizerReductionType reductionType_,
                                 DataType accumType_,
                                 DataType accl1Type_,
                                 DataType accl2Type_,
                                 DataType accl3Type_,
                                 const Op::Settings &settings_)
    : VarUpdateWithUpdaterOp(Onnx::CustomOperators::AdaptiveCombo, settings_),
      initLr(initialLr), initWd(initialWd), initA(initialA), initM(initialM),
      initEps(initialEps), initLs(initialLs), initGs(initialGs), mode(mode_),
      decayMode(decayMode_), withGradAccum(withGradAccum_),
      reductionType(reductionType_), accumType(accumType_),
      accl1Type(accl1Type_), accl2Type(accl2Type_), accl3Type(accl3Type_) {}

void AdaptiveComboOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  if (initLr.isConst()) {
    os.appendAttribute("const learning rate", initLr.val());
  }

  if (initWd.isConst()) {
    os.appendAttribute("const weight decay", initWd.val());
  }

  if (initA.isConst()) {
    os.appendAttribute("const alpha", initA.val());
  }

  if (initEps.isConst()) {
    os.appendAttribute("const eps", initEps.val());
  }

  if (initLs.isConst()) {
    os.appendAttribute("const loss scaling", initLs.val());
  }

  os.appendAttribute("reduction type", static_cast<int>(reductionType));
  os.appendAttribute("adaptive mode", static_cast<int>(mode));
  os.appendAttribute("decay mode", static_cast<int>(decayMode));
}

std::unique_ptr<Op> AdaptiveComboOp::clone() const {
  return std::make_unique<AdaptiveComboOp>(*this);
}

std::map<InIndex, TensorId> AdaptiveComboOp::optimizerInputs() const {

  std::map<InIndex, TensorId> m;

  if (!initLr.isConst()) {
    auto index = getLrInIndex();
    m.insert({index, inId(index)});
  }

  if (!initWd.isConst()) {
    auto index = getWdInIndex();
    m.insert({index, inId(index)});
  }

  if (!initA.isConst()) {
    auto index = getAlphaInIndex();
    m.insert({index, inId(index)});
  }

  if (!initM.isConst()) {
    auto index = getMomentumInIndex();
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

  if (!initGs.isConst()) {
    auto index = getGsInIndex();
    m.insert({index, inId(index)});
  }

  return m;
}

std::set<InIndex> AdaptiveComboOp::optionalInputs() const {
  return {getLrInIndex(),
          getWdInIndex(),
          getAlphaInIndex(),
          getMomentumInIndex(),
          getEpsInIndex(),
          getLsInIndex(),
          getGsInIndex()};
}

} // namespace popart
