#ifndef GUARD_NEURALNET_OPMANAGER_HPP
#define GUARD_NEURALNET_OPMANAGER_HPP

#include <functional>
#include <map>
#include <vector>
#include <poponnx/attributes.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op.hpp>

namespace poponnx {

class OpManager {

public:
  using OpFactoryFunc =
      std::function<std::unique_ptr<Op>(const OperatorIdentifier &_opid,
                                        Ir *_ir,
                                        const std::string &name,
                                        const Attributes &_attr)>;

  struct OpInfo {
    OpInfo() : isPublic(false), f1(nullptr) {}
    // Does poponnx expose the Op in its public API ?
    bool isPublic;
    OpFactoryFunc f1;
    // TBD could hold meta information about input/ouput/attributes expected
  };

  OpManager() = default;

public:
  // Registration method
  static void
  registerOp(const OperatorIdentifier &opid, bool isPublic, OpFactoryFunc func);

  // Factory creation method
  static std::unique_ptr<Op> createOp(const OperatorIdentifier &opid,
                                      Ir *ir,
                                      const std::string &name = "",
                                      const Attributes &_attr = {});

  // Get the list of registered op's, should this return an OperatorIdentifier
  static const std::vector<OperatorIdentifier>
  getSupportedOperations(bool includePrivate);

private:
  // Singleton
  static OpManager &getInstance();

  // Map of registered ops
  std::map<OperatorIdentifier, OpInfo, OperatorIdentifierLess> opMap;
};

// This class registers a lambda function to create a op with the
// OpManager
template <class OP> class OpCreator {
public:
  OpCreator(const OperatorIdentifier &opid, bool isPublic = true) {
    OpManager::registerOp(
        opid,
        isPublic,
        [](const OperatorIdentifier &_opid,
           Ir *ir,
           const std::string &_name = "",
           const Attributes &attr   = {}) -> std::unique_ptr<Op> {
          return std::unique_ptr<OP>(new OP(_opid, ir, _name, attr));
        });
  }

  OpCreator(const OperatorIdentifier &opid,
            OpManager::OpFactoryFunc func,
            bool isPublic = true) {
    OpManager::registerOp(opid, isPublic, func);
  }
};

template <class OP> class LossOpCreator {
public:
  LossOpCreator(const OperatorIdentifier &opid, bool isPublic = false) {
    OpManager::registerOp(opid, isPublic, nullptr);
  }
};

template <class GRADOP> class GradOpCreator {
public:
  GradOpCreator(const OperatorIdentifier &opid, bool isPublic = false) {
    OpManager::registerOp(opid, isPublic, nullptr);
  }
};

} // namespace poponnx

#endif
