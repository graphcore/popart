#ifndef GUARD_NEURALNET_POPOPXMANAGER_HPP
#define GUARD_NEURALNET_POPOPXMANAGER_HPP

#include <poponnx/error.hpp>
#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class OpxManager {

  OpxManager() = default;

  // Singleton
  static OpxManager &getInstance();

public:
  using OpxFactoryFunc =
      std::function<std::unique_ptr<Opx>(Op *op, Devicex *devicex)>;

  static void registerOpx(const OperatorIdentifier &opid, OpxFactoryFunc func);

  static std::unique_ptr<Opx> createOpx(Op *op, Devicex *devicex);

private:
  std::map<std::reference_wrapper<const OperatorIdentifier>,
           OpxFactoryFunc,
           OperatorIdentifierLess>
      factory;
};

template <class OPX> class OpxCreator {
public:
  OpxCreator(const OperatorIdentifier &opid) {
    OpxManager::registerOpx(
        opid, [](Op *op, Devicex *devicex) -> std::unique_ptr<Opx> {
          return std::unique_ptr<OPX>(new OPX(op, devicex));
        });
  }

  OpxCreator(const OperatorIdentifier &opid, std::string errMsg) {
    OpxManager::registerOpx(opid,
                            [errMsg](Op *, Devicex *) -> std::unique_ptr<Opx> {
                              throw error(errMsg);
                            });
  }
};

} // namespace popx
} // namespace poponnx

#endif
