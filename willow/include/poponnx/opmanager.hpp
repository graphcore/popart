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
                                        const Op::Settings &settings,
                                        const Attributes &_attr)>;

  struct OpInfo {
    OpInfo(const OperatorIdentifier &_id)
        : isPublic(false), id(_id), f1(nullptr) {}
    // Does poponnx expose the Op in its public API ?
    bool isPublic;
    const OperatorIdentifier id;
    OpFactoryFunc f1;
  };

  OpManager() = default;

public:
  // Registration method
  static void
  registerOp(const OperatorIdentifier &opid, bool isPublic, OpFactoryFunc func);

  // Factory creation method
  // creates a op with matches the domain/type and has the largest version <=
  // opsetVersion
  static std::unique_ptr<Op> createOp(const OpDomain &domain,
                                      const OpType &type,
                                      const int opsetVersion,
                                      Graph &graph,
                                      const std::string &name = "",
                                      const Scope &scope      = {},
                                      const Attributes &_attr = {});

  // creates a op with matches the opid
  static std::unique_ptr<Op> createOp(const OperatorIdentifier &opid,
                                      Graph &graph,
                                      const std::string &name = "",
                                      const Attributes &_attr = {});

  // Get the list of registered op's, should this return an OperatorIdentifier
  static const std::vector<OperatorIdentifier>
  getSupportedOperations(bool includePrivate);

  static OpVersion getOpVersionFromOpSet(const OpDomain &opDomain,
                                         const OpType &type,
                                         const int opsetVersion);

private:
  std::unique_ptr<Op> create(const OperatorIdentifier &opid,
                             Graph &graph,
                             const std::string &name,
                             const Scope &scope,
                             const Attributes &_attr,
                             OpFactoryFunc func);
  // Singleton
  static OpManager &getInstance();

  // Map of domain/type to the list of supported versions and the opInfo
  std::map<std::pair<OpDomain, OpType>, std::map<int, OpInfo>> opMap;
};

// This class registers a lambda function to create a op with the
// OpManager

template <class OP> class OpCreator {

  void registerOp(const OperatorIdentifier &opid, bool isPublic) {
    OpManager::registerOp(opid,
                          isPublic,
                          [](const OperatorIdentifier &_opid,
                             const Op::Settings &settings,
                             const Attributes &) -> std::unique_ptr<Op> {
                            return std::unique_ptr<OP>(new OP(_opid, settings));
                          });
  }

public:
  OpCreator(const OperatorIdentifier &opid, bool isPublic = true) {
    registerOp(opid, isPublic);
  }

  OpCreator(const std::vector<OperatorIdentifier> &opids,
            bool isPublic = true) {
    for (const auto &opid : opids) {
      registerOp(opid, isPublic);
    }
  }

  OpCreator(const OperatorIdentifier &opid,
            OpManager::OpFactoryFunc func,
            bool isPublic = true) {
    OpManager::registerOp(opid, isPublic, func);
  }

  OpCreator(const std::vector<OperatorIdentifier> &opids,
            OpManager::OpFactoryFunc func,
            bool isPublic = true) {
    for (const auto &opid : opids) {
      OpManager::registerOp(opid, isPublic, func);
    }
  }
};

} // namespace poponnx

#endif
