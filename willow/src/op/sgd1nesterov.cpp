// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <popart/op/sgd1nesterov.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/region.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

SGD1NesterovOp::SGD1NesterovOp(const OperatorIdentifier &_opid,
                               float initInverseLossScale_,
                               float initWd_,
                               float initNgsf_,
                               float initMm_,
                               const Op::Settings &settings_)
    : Op(_opid, settings_), initInverseLossScale(initInverseLossScale_),
      initWd(initWd_), initNgsf(initNgsf_), initMm(initMm_) {}

std::unique_ptr<Op> SGD1NesterovOp::clone() const {
  return std::make_unique<SGD1NesterovOp>(*this);
}

void SGD1NesterovOp::setup() {
  outInfo(getOutIndex()) = inInfo(getGradInIndex());
}

void SGD1NesterovOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("initInverseLossScale", initInverseLossScale);
  os.appendAttribute("initWd", initWd);
  os.appendAttribute("initNgsf", initNgsf);
  os.appendAttribute("initMm", initMm);
}

} // namespace popart
