// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OPMANAGER_HPP
#define GUARD_NEURALNET_OPMANAGER_HPP

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <popart/attributes.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/tensordata.hpp> // IWYU pragma: keep
#include <popart/tensorinfo.hpp>

#include "popart/basicoptionals.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/half.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {
class Graph;
class Scope;
class Tensor;
class any;

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
                const std::vector<TensorId> &_inputIds,
                const std::vector<TensorId> &_outputIds)
      : opid(_opid), settings(_settings), attributes(_attributes),
        inputIds(_inputIds), outputIds(_outputIds) {}

  bool hasInputIds() const { return !inputIds.empty(); }
  bool hasOutputIds() const { return !outputIds.empty(); }

  const std::vector<TensorId> &getInputIds() const;
  const std::vector<TensorId> &getOutputIds() const;
  Tensor *getInputTensor(int index) const;
  TensorData *getInputTensorData(int index) const;
  TensorInfo &getInputTensorInfo(int index) const;
  bool hasInputTensor(int index) const;
  std::string debugName() const;

  template <typename T>
  std::vector<T> getInputData(int index,
                              const std::set<DataType> &acceptedTypes) const {
    auto &tensorInfo = getInputTensorInfo(index);
    if (acceptedTypes.find(tensorInfo.dataType()) == acceptedTypes.end()) {
      throw error("Trying to get data from tensor of type {}, but the "
                  "acceptable types are {}",
                  tensorInfo.dataType(),
                  acceptedTypes);
    }

    switch (tensorInfo.dataType()) {
    case DataType::INT32: {
      auto x = getInputData<int32_t>(index);
      return std::vector<T>(x.begin(), x.end());
    }
    case DataType::INT64: {
      auto x = getInputData<int64_t>(index);
      return std::vector<T>(x.begin(), x.end());
    }
    case DataType::FLOAT: {
      auto x = getInputData<float>(index);
      return std::vector<T>(x.begin(), x.end());
    }
    case DataType::FLOAT16: {
      auto x = getInputData<float16_t>(index);
      return std::vector<T>(x.begin(), x.end());
    }
    default:
      throw internal_error("Unhandled type {}.", tensorInfo.dataType());
    }
  }

  template <typename T> std::vector<T> getInputData(int index) const {
    auto tensorData  = getInputTensorData(index);
    auto &tensorInfo = getInputTensorInfo(index);

    if (tensorInfo.rank() > 1) {
      throw error("Can only return data for rank 0 or rank 1 tensors. Tensor "
                  "is of rank {}",
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

  // Gets a single scalar value from the input at `index`.
  // This will throw an error if:
  //   * The input does not exist.
  //   * The input tensor is not rank 1.
  //   * The input tensor does not contains exactly 1 value.
  template <typename T> T getInputScalarValue(int index) const {
    std::vector<T> values = getInputData<T>(index);
    if (values.size() != 1) {
      throw error(
          "Expected input at index {} to has a shape of [1]. It has shape [{}]",
          index,
          values.size());
    }
    return values.at(0);
  }

  // Gets a single scalar value from the input at `index`, or return the default
  // value if the input is not present. This will throw an error if:
  //   * The input tensor is not rank 1.
  //   * The input tensor does not contains exactly 1 value.
  template <typename T> T getInputScalarValue(int index, T defaultValue) const {
    if (inputIds.size() > index && inputIds.at(index) != "") {
      return getInputScalarValue<T>(index);
    } else {
      return defaultValue;
    }
  }

private:
  const std::vector<TensorId> &inputIds;
  const std::vector<TensorId> &outputIds;
};

class OpManager {

public:
  // The basic op factory function is responsible for creating and returning a
  // unique_ptr to an instance of an op.
  using OpFactoryFunc =
      std::function<std::unique_ptr<Op>(const OpCreatorInfo &)>;

  // The complex op factory function is responsible for creating an op, adding
  // it to the graph, and connecting the inputs and outputs. Graph could have
  // been an attribute of OpCreatorInfo, but having the two factory funcs with
  // the same arguments and difference return types was causing errors like
  // "call to constructor of 'OpCreator<...>' is ambiguous".
  using ComplexOpFactoryFunc =
      std::function<Op *(const OpCreatorInfo &, Graph &graph)>;

  class OpInfo {
  public:
    OpInfo(const OperatorIdentifier &_id,
           bool _isPublic,
           const OpDefinition &_details,
           OpFactoryFunc _f1)
        : isPublic(_isPublic), id(_id), details(_details), simpleFactory(_f1),
          complexFactory(BasicOptional<ComplexOpFactoryFunc>()) {}

    OpInfo(const OperatorIdentifier &_id,
           bool _isPublic,
           const OpDefinition &_details,
           ComplexOpFactoryFunc _f2)
        : isPublic(_isPublic), id(_id), details(_details),
          simpleFactory(BasicOptional<OpFactoryFunc>()), complexFactory(_f2) {}

    // Does popart expose the Op in its public API ?
    bool isPublic;
    const OperatorIdentifier id;
    OpDefinition details;

    OpFactoryFunc &getSimpleFactory();
    ComplexOpFactoryFunc &getComplexFactory();
    bool hasComplexFactory();

  private:
    BasicOptional<OpFactoryFunc> simpleFactory;
    BasicOptional<ComplexOpFactoryFunc> complexFactory;
  };

  OpManager() = default;

public:
  static void registerOp(const OpInfo &opInfo);

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
           const std::string &name                = "",
           const Scope &scope                     = {},
           const Attributes &_attr                = {},
           const std::vector<TensorId> &inputIds  = {},
           const std::vector<TensorId> &outputIds = {});

  // creates a op with matches the opid
  static std::unique_ptr<Op> createOp(const OperatorIdentifier &opid,
                                      Graph &graph,
                                      const std::string &name = "",
                                      const Attributes &_attr = {});

  static std::unique_ptr<Op>
  createOpWithInputs(const OperatorIdentifier &opid,
                     Graph &graph,
                     const std::string &name,
                     const Attributes &_attr,
                     const std::vector<TensorId> &inIds);

  // Creates an op from the onnx node and adds it to the graph.
  static Op *createOpInGraph(const Node &node, Graph &graph);

  // Get the list of registered op's, should this return an OperatorIdentifier
  static const std::vector<OperatorIdentifier>
  getSupportedOperations(bool includePrivate);

  static const std::vector<OperatorIdentifier>
  getUnsupportedOperations(int opsetVersion);

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
                             const std::vector<TensorId> &outputIds,
                             OpFactoryFunc func);

  Op *create(const OperatorIdentifier &opid,
             Graph &graph,
             const std::string &name,
             const Scope &scope,
             const Attributes &_attr,
             const std::vector<TensorId> &inputIds,
             const std::vector<TensorId> &outputIds,
             ComplexOpFactoryFunc func);

  // Singleton
  static OpManager &getInstance();

  // Search opMap for the OpInfo with the highest version that is less than or
  // equal to the opsetVersion. This method will return null if a suitable op is
  // not found.
  OpInfo *
  findOpInfo(const OpDomain &domain, const OpType &type, int opsetVersion);

  // Check the version of `opInfo` and make sure it matches the version of the
  // op found in the onnx opset of `opsetVersion`.
  void checkOpVersionAgainstOpset(const OpInfo *opInfo,
                                  int opsetVersion,
                                  Graph &graph);

  // Map of domain/type to the list of supported versions and the opInfo
  std::map<std::pair<OpDomain, OpType>, std::map<int, OpInfo>> opMap;
};

// This class registers a lambda function to create a op with the
// OpManager

template <class OP> class OpCreator {
public:
  OpCreator(const OpDefinitions &opDefinitions, bool isPublic = true) {
    for (const auto &version : opDefinitions) {
      OpManager::OpFactoryFunc func =
          [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
        return std::unique_ptr<OP>(new OP(info.opid, info.settings));
      };
      OpManager::registerOp({version.first, isPublic, version.second, func});
    }
  }

  OpCreator(const OpDefinitions &opDefinitions,
            OpManager::OpFactoryFunc func,
            bool isPublic = true) {
    for (const auto &version : opDefinitions) {
      OpManager::registerOp({version.first, isPublic, version.second, func});
    }
  }

  OpCreator(const OpDefinitions &opDefinitions,
            OpManager::ComplexOpFactoryFunc func,
            bool isPublic = true) {
    for (const auto &version : opDefinitions) {
      OpManager::registerOp({version.first, isPublic, version.second, func});
    }
  }
};

} // namespace popart

#endif
