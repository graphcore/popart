
#include <poponnx/opmanager.hpp>

namespace poponnx {

std::ostream &operator<<(std::ostream &os, const OperatorIdentifier &opid) {
  os << opid.domain << "." << opid.type << ":" << opid.version;
  return os;
}

OpManager &OpManager::getInstance() {
  static OpManager instance;
  return instance;
}

void OpManager::registerOp(const OperatorIdentifier &opid,
                           bool isPublic,
                           OpFactoryFunc func) {

  OpInfo info;
  info.isPublic = isPublic;
  info.f1       = func;

  getInstance().opMap.emplace(
      std::pair<OperatorIdentifier, OpInfo>(opid, info));
}

const std::vector<OperatorIdentifier>
OpManager::getSupportedOperations(bool includePrivate) {
  std::vector<OperatorIdentifier> list;

  for (auto &op : OpManager::getInstance().opMap) {
    if (op.second.isPublic || includePrivate) {
      list.push_back(op.first);
    }
  }

  return list;
}

std::unique_ptr<Op> OpManager::createOp(const OperatorIdentifier &opid,
                                        Ir *ir,
                                        const std::string &name,
                                        const Attributes &attr) {

  OpManager &self = getInstance();

  auto it = self.opMap.find(opid);
  if (it != self.opMap.end()) {
    return it->second.f1(opid, ir, name, attr);
  } else {
    return nullptr;
  }
}

} // namespace poponnx
