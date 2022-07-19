// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OPXMANAGER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OPXMANAGER_HPP_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/error.hpp>

#include "popart/operatoridentifier.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;
class PopOpx;

class OpxManager {

  OpxManager() = default;

  // Singleton
  static OpxManager &getInstance();

public:
  using OpxFactoryFunc =
      std::function<std::unique_ptr<PopOpx>(Op *op, Devicex *devicex)>;

  static void registerOpx(const OperatorIdentifier &opid, OpxFactoryFunc func);

  static std::unique_ptr<PopOpx> createOpx(Op *op, Devicex *devicex);

private:
  std::map<OperatorIdentifier, OpxFactoryFunc, OperatorIdentifierLess> factory;
};

template <class OPX> class OpxCreator {
public:
  OpxCreator(const OperatorIdentifier &opid) {
    OpxManager::registerOpx(
        opid, [](Op *op, Devicex *devicex) -> std::unique_ptr<PopOpx> {
          return std::unique_ptr<OPX>(new OPX(op, devicex));
        });
  }

  OpxCreator(const std::vector<OperatorIdentifier> &opids) {
    for (const auto &opid : opids) {
      OpxManager::registerOpx(
          opid, [](Op *op, Devicex *devicex) -> std::unique_ptr<PopOpx> {
            return std::unique_ptr<OPX>(new OPX(op, devicex));
          });
    }
  }

  OpxCreator(const OperatorIdentifier &opid, std::string errMsg) {
    OpxManager::registerOpx(
        opid, [errMsg](Op *, Devicex *) -> std::unique_ptr<PopOpx> {
          throw error(errMsg);
        });
  }

  OpxCreator(const std::vector<OperatorIdentifier> &opids, std::string errMsg) {
    for (const auto &opid : opids) {
      OpxManager::registerOpx(
          opid, [errMsg](Op *, Devicex *) -> std::unique_ptr<PopOpx> {
            throw error(errMsg);
          });
    }
  }

  OpxCreator(const OperatorIdentifier &opid, OpxManager::OpxFactoryFunc func) {
    OpxManager::registerOpx(opid, func);
  }

  OpxCreator(const std::vector<OperatorIdentifier> &opids,
             OpxManager::OpxFactoryFunc func) {
    for (const auto &opid : opids) {
      OpxManager::registerOpx(opid, func);
    }
  }
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OPXMANAGER_HPP_
