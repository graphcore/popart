// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <popart/op.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/popx/popopx.hpp"

namespace popart {
namespace popx {
class Devicex;

OpxManager &OpxManager::getInstance() {
  static OpxManager instance;
  return instance;
}

void OpxManager::registerOpx(const OperatorIdentifier &opid,
                             OpxFactoryFunc func) {
  getInstance().factory.emplace(
      std::pair<OperatorIdentifier, OpxFactoryFunc>(opid, func));
}

std::unique_ptr<PopOpx> OpxManager::createOpx(Op *op, Devicex *devicex) {

  OpxManager &self = getInstance();
  auto it          = self.factory.find(op->opid);
  if (it != self.factory.end()) {
    return it->second(op, devicex);
  } else {
    return nullptr;
  }
}

} // namespace popx
} // namespace popart
