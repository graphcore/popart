// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <map>
#include <set>
#include <string>
#include <utility>
#include <popart/op/sgdcombobase.hpp>
#include <popart/opserialiser.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
struct OperatorIdentifier;

SGDMComboBaseOp::SGDMComboBaseOp(const OperatorIdentifier &opid,
                                 OptimizerValue initialSmm1,
                                 OptimizerValue initialDpsf1,
                                 OptimizerValue initialSwd1,
                                 OptimizerValue initialSlr1,
                                 OptimizerReductionType reductionType_,
                                 const Op::Settings &settings_)
    : VarUpdateWithUpdaterOp(opid, settings_), initSmm1(std::move(initialSmm1)),
      initDpsf1(std::move(initialDpsf1)), initSwd1(std::move(initialSwd1)),
      initSlr1(std::move(initialSlr1)), reductionType(reductionType_),
      nesterov(false) {}

SGDMComboBaseOp::SGDMComboBaseOp(const OperatorIdentifier &opid,
                                 OptimizerValue initialSmm1,
                                 OptimizerValue initialDpsf1,
                                 OptimizerValue initialSwd1,
                                 OptimizerValue initialSlr1,
                                 OptimizerValue initialMm,
                                 OptimizerValue initialWd,
                                 OptimizerValue initialNgsf,
                                 OptimizerValue initialNdsf,
                                 OptimizerReductionType reductionType_,
                                 const Op::Settings &settings_)
    : VarUpdateWithUpdaterOp(opid, settings_), initSmm1(std::move(initialSmm1)),
      initDpsf1(std::move(initialDpsf1)), initSwd1(std::move(initialSwd1)),
      initSlr1(std::move(initialSlr1)), initMm(std::move(initialMm)),
      initWd(std::move(initialWd)), initNgsf(std::move(initialNgsf)),
      initNdsf(std::move(initialNdsf)), reductionType(reductionType_),
      nesterov(true) {}

void SGDMComboBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  if (initSmm1.isConst()) {
    os.appendAttribute("const scaled momentum", initSmm1.val());
  }

  if (initDpsf1.isConst()) {
    os.appendAttribute("const dampening scale factor", initDpsf1.val());
  }

  if (initSwd1.isConst()) {
    os.appendAttribute("const weight decay scale factor", initSwd1.val());
  }

  if (initSlr1.isConst()) {
    os.appendAttribute("const scaled learning rate", initSlr1.val());
  }

  os.appendAttribute("reduction type", static_cast<int>(reductionType));
}

std::map<InIndex, TensorId> SGDMComboBaseOp::optimizerInputs() const {

  std::map<InIndex, TensorId> m;

  if (!initSlr1.isConst()) {
    auto index = getSlr1InIndex();
    m.insert({index, inId(index)});
  }

  if (!initSwd1.isConst()) {
    auto index = getSwd1InIndex();
    m.insert({index, inId(index)});
  }

  if (!initSmm1.isConst()) {
    auto index = getSmm1InIndex();
    m.insert({index, inId(index)});
  }

  if (!initDpsf1.isConst()) {
    auto index = getDpsf1InIndex();
    m.insert({index, inId(index)});
  }

  if (nesterov) {
    if (!initMm.isConst()) {
      auto index = getMmInIndex();
      m.insert({index, inId(index)});
    }

    if (!initWd.isConst()) {
      auto index = getWdInIndex();
      m.insert({index, inId(index)});
    }

    if (!initNgsf.isConst()) {
      auto index = getNgsfInIndex();
      m.insert({index, inId(index)});
    }

    if (!initNdsf.isConst()) {
      auto index = getNdsfInIndex();
      m.insert({index, inId(index)});
    }
  }

  return m;
}

std::set<InIndex> SGDMComboBaseOp::optionalInputs() const {
  return {getSmm1InIndex(),
          getDpsf1InIndex(),
          getSwd1InIndex(),
          getSlr1InIndex(),
          getMmInIndex(),
          getWdInIndex(),
          getNgsfInIndex(),
          getNdsfInIndex()};
}

} // namespace popart
