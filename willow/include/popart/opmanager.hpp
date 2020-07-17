// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OPMANAGER_HPP
#define GUARD_NEURALNET_OPMANAGER_HPP

#include <functional>
#include <map>
#include <vector>
#include <popart/attributes.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/vendored/any.hpp>

#include <popart/tensordata.hpp>
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

class OpCreatorInfo {
public:
  const OperatorIdentifier &opid;
  const Op::Settings &settings;
  const Attributes &attributes;

  OpCreatorInfo(const OperatorIdentifier &_opid,
                const Op::Settings &_settings,
                const Attributes &_attributes,
                const std::vector<TensorId> &_inputIds)
      : opid(_opid), settings(_settings), attributes(_attributes),
        inputIds(_inputIds) {}

  const std::vector<TensorId> &getInputIds() const;
  Tensor *getInputTensor(int index) const;
  TensorData *getInputTensorData(int index) const;
  TensorInfo &getInputTensorInfo(int index) const;

  template <typename T> std::vector<T> getInputData(int index) const {
    auto tensorData  = getInputTensorData(index);
    auto &tensorInfo = getInputTensorInfo(index);

    if (tensorInfo.rank() != 1) {
      throw error(
          "Can only return data for rank 1 tensors. Tensor is of rank {}",
          tensorInfo.rank());
    }

    if (getDataType<T>() == tensorInfo.dataType()) {
      return tensorData->copyDataAs<T>(tensorInfo.nelms());
    } else {
      throw error("Trying to get data as incorrect type. Trying to get data as "
                  "{} but it is of type {}",
                  getDataType<T>(),
                  tensorInfo.dataType());
    }
  }

private:
  const std::vector<TensorId> &inputIds;
};

class OpManager {

public:
  using OpFactoryFunc =
      std::function<std::unique_ptr<Op>(const OpCreatorInfo &)>;

#ifndef DEPRECATE_LEGACY_OP_FACTORY
  using LegacyOpFactoryFunc =
      std::function<std::unique_ptr<Op>(const OperatorIdentifier &_opid,
                                        const Op::Settings &settings,
                                        const Attributes &_attr)>;
#endif

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

  static Attributes
  getAttributesFromAnyMap(std::map<std::string, popart::any> attributes);

  // Factory creation method
  // creates a op with matches the domain/type and has the largest version <=
  // opsetVersion
  static std::unique_ptr<Op>
  createOp(const OpDomain &domain,
           const OpType &type,
           const int opsetVersion,
           Graph &graph,
           const std::string &name               = "",
           const Scope &scope                    = {},
           const Attributes &_attr               = {},
           const std::vector<TensorId> &inputIds = {});

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
                             const std::vector<TensorId> &inputIds,
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
                          [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
                            return std::unique_ptr<OP>(
                                new OP(info.opid, info.settings));
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

#ifndef DEPRECATE_LEGACY_OP_FACTORY
  OpCreator(const OpDefinitions &opDefinitions,
            OpManager::LegacyOpFactoryFunc func,
            bool isPublic = true) {
    OpManager::OpFactoryFunc wrapper = [func](const OpCreatorInfo &info) {
      // Adding this warning when the function is called, rather than when the
      // function is registered, ensures that logging has been set up.
      logging::warn("You are using a deprecated function signature for the "
                    "factory function of {}. This will be removed in a future "
                    "release. Please update it to use the signature "
                    "`std::unique_ptr<Op> (const OpCreatorInfo &)`",
                    info.opid.type);
      return func(info.opid, info.settings, info.attributes);
    };

    for (const auto &version : opDefinitions) {
      OpManager::registerOp(version.first, version.second, isPublic, wrapper);
    }
  }
#endif
};

} // namespace popart

#endif
