#include <poponnx/error.hpp>
#include <poponnx/op.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

OpxManager &OpxManager::getInstance() {
  static OpxManager instance;
  return instance;
}

void OpxManager::registerOpx(const OperatorIdentifier &opid,
                             OpxFactoryFunc func) {
  getInstance().factory.emplace(
      std::pair<std::reference_wrapper<const OperatorIdentifier>,
                OpxFactoryFunc>(opid, func));
}

std::unique_ptr<Opx> OpxManager::createOpx(Op *op, Devicex *devicex) {

  OpxManager &self = getInstance();
  auto it          = self.factory.find(op->opid);
  if (it != self.factory.end()) {
    return it->second(op, devicex);
  } else {
    return nullptr;
  }
}

} // namespace popx
} // namespace poponnx
