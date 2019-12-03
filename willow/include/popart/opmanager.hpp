#ifndef GUARD_NEURALNET_OPMANAGER_HPP
#define GUARD_NEURALNET_OPMANAGER_HPP

#include <functional>
#include <map>
#include <vector>
#include <popart/attributes.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>

#include <popart/tensorinfo.hpp>

namespace popart {

class OpDefinition {
public:
  struct Input {
    std::string name;
    std::vector<DataType> supportedTensors;
    bool constant;

    Input(std::string n, std::vector<DataType> t, bool _constant = false)
        : name(n), supportedTensors(t), constant(_constant) {}
  };
  struct Output {
    std::string name;
    std::vector<DataType> supportedTensors;

    Output(std::string n, std::vector<DataType> t)
        : name(n), supportedTensors(t) {}
  };
  struct Attribute {
    std::string supportedValuesRegex;

    Attribute(std::string regex) : supportedValuesRegex(regex) {}
  };

  using DataTypes  = std::vector<DataType>;
  using Inputs     = std::vector<Input>;
  using Outputs    = std::vector<Output>;
  using Attributes = std::map<std::string, Attribute>;

  Inputs inputs;
  Outputs outputs;
  Attributes attributes;

  OpDefinition() {}

  OpDefinition(Inputs i, Outputs o, Attributes a)
      : inputs(i), outputs(o), attributes(a) {}
};

using OpDefinitions =
    std::map<OperatorIdentifier, OpDefinition, OperatorIdentifierLess>;

using OpIdentifierList = std::vector<OperatorIdentifier>;

std::ostream &operator<<(std::ostream &os,
                         const std::vector<DataType> &dataTypes);

class OpManager {

public:
  using OpFactoryFunc =
      std::function<std::unique_ptr<Op>(const OperatorIdentifier &_opid,
                                        const Op::Settings &settings,
                                        const Attributes &_attr)>;

  struct OpInfo {
    OpInfo(const OperatorIdentifier &_id)
        : isPublic(false), id(_id), f1(nullptr), details({}) {}
    // Does popart expose the Op in its public API ?
    bool isPublic;
    const OperatorIdentifier id;
    OpFactoryFunc f1;
    OpDefinition details;
  };

  OpManager() = default;

public:
  // Registration method
  static void registerOp(const OperatorIdentifier &opid,
                         const OpDefinition &details,
                         bool isPublic,
                         OpFactoryFunc func);

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

  static const OpDefinitions
  getSupportedOperationsDefinition(bool includePrivate);

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

  void registerOp(const OperatorIdentifier &opid,
                  const OpDefinition &details,
                  bool isPublic) {
    OpManager::registerOp(opid,
                          details,
                          isPublic,
                          [](const OperatorIdentifier &_opid,
                             const Op::Settings &settings,
                             const Attributes &) -> std::unique_ptr<Op> {
                            return std::unique_ptr<OP>(new OP(_opid, settings));
                          });
  }

public:
  OpCreator(const OpDefinitions &opDefinitions, bool isPublic = true) {
    for (const auto &version : opDefinitions) {
      registerOp(version.first, version.second, isPublic);
    }
  }

  OpCreator(const OpDefinitions &opDefinitions,
            OpManager::OpFactoryFunc func,
            bool isPublic = true) {
    for (const auto &version : opDefinitions) {
      OpManager::registerOp(version.first, version.second, isPublic, func);
    }
  }
};

} // namespace popart

#endif
