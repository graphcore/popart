// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPOPXMANAGER_HPP
#define GUARD_NEURALNET_POPOPXMANAGER_HPP

#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

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

#endif
