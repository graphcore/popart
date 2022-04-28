// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <parsedtensorid.hpp>
#include <set>
#include <string>
#include <typeinfo>
#include <vector>
#include <poprithms/util/printiter.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/optimizer.hpp>
#include <popart/scope.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/graph.hpp"
#include "popart/graphid.hpp"
#include "popart/half.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensors.hpp"
#include "popart/vendored/any.hpp"
#include "popart/vendored/optional.hpp"

namespace std {

std::ostream &operator<<(std::ostream &ss, const popart::any &value) {
  const std::type_info &valueType = value.type();

  if (valueType == typeid(float)) {
    ss << popart::any_cast<float>(value);
  } else if (valueType == typeid(double)) {
    ss << popart::any_cast<double>(value);
  } else if (valueType == typeid(int)) {
    ss << popart::any_cast<int>(value);
  } else if (valueType == typeid(int64_t)) {
    ss << popart::any_cast<int64_t>(value);
  } else if (valueType == typeid(uint32_t)) {
    ss << popart::any_cast<uint32_t>(value);
  } else if (valueType == typeid(uint64_t)) {
    ss << popart::any_cast<uint64_t>(value);
  } else if (valueType == typeid(std::string)) {
    ss << popart::any_cast<std::string>(value);
  } else if (valueType == typeid(std::vector<float>)) {
    ss << popart::any_cast<std::vector<float>>(value);
  } else if (valueType == typeid(std::vector<double>)) {
    ss << popart::any_cast<std::vector<double>>(value);
  } else if (valueType == typeid(std::vector<int64_t>)) {
    ss << popart::any_cast<std::vector<int64_t>>(value);
  } else if (valueType == typeid(popart::Scope)) {
    ss << popart::any_cast<popart::Scope>(value);
  } else if (valueType == typeid(bool)) {
    ss << popart::any_cast<bool>(value);
  } else if (valueType == typeid(nonstd::optional<int64_t>)) {
    ss << popart::any_cast<nonstd::optional<int64_t>>(value);
  } else if (valueType == typeid(nonstd::optional<float>)) {
    ss << popart::any_cast<nonstd::optional<float>>(value);
  } else if (valueType == typeid(nonstd::optional<double>)) {
    ss << popart::any_cast<nonstd::optional<double>>(value);
  } else if (valueType == typeid(std::map<popart::TensorId, uint64_t>)) {
    ss << popart::any_cast<std::map<popart::TensorId, uint64_t>>(value);
  } else {
    throw popart::error("Unsupported popart::any type for operator<< ({})",
                        valueType.name());
  }
  return ss;
}
} // namespace std

namespace popart {

namespace {

bool hasConnectedTensor(const Graph &graph, const TensorId id) {
  if (graph.getTensors().contains(id)) {
    return graph.getTensors().get(id)->consumers.getTotal() > 0;
  } else {
    return false;
  }
}

} // namespace

bool hasSingleConnectedLossScaleTensor(const Graph &graph) {
  const Optimizer &optimizer = graph.getIr().getOptimizer();
  TensorId lsFP16 = optimizer.getLossScalingTensorId(DataType::FLOAT16);
  TensorId lsFP32 = optimizer.getLossScalingTensorId(DataType::FLOAT);

  // Exclusive or
  return hasConnectedTensor(graph, lsFP16) != hasConnectedTensor(graph, lsFP32);
}

Tensor *getLossScaleTensor(const Graph &graph) {
  const Ir &ir               = graph.getIr();
  const Optimizer &optimizer = ir.getOptimizer();

  TensorId lsFP16 = optimizer.getLossScalingTensorId(DataType::FLOAT16);
  TensorId lsFP32 = optimizer.getLossScalingTensorId(DataType::FLOAT);
  bool existsLossScaleFP16 = hasConnectedTensor(graph, lsFP16);
  bool existsLossScaleFP32 = hasConnectedTensor(graph, lsFP32);

  Tensor *lossScaleTensor;
  if (existsLossScaleFP16 && existsLossScaleFP32) {
    throw error("Unable to determine the data type of the loss scale tensor, "
                "as both tensors '{}' and '{}' exist in graph {}",
                lsFP16,
                lsFP32,
                graph.id);
  } else {
    if (existsLossScaleFP16) {
      lossScaleTensor = graph.getTensors().get(lsFP16);
    } else if (existsLossScaleFP32) {
      lossScaleTensor = graph.getTensors().get(lsFP32);
    } else {
      throw error("Unable to find any loss scale tensor in graph '{}'",
                  graph.id);
    }
  }

  return lossScaleTensor;
}

std::set<Tensor *> getInverseLossScaleTensors(const Graph &graph) {
  const Ir &ir               = graph.getIr();
  const Optimizer &optimizer = ir.getOptimizer();

  // To ensure that the tensor we return from this method is the compound
  // scalar this is used to apply the inverse loss scale in all VarUpdateOps
  // in this graph, we check that all Variable tensors have the same type.
  // Otherwise the graph will contain more than one of these tensors; one
  // per type.
  auto variables = graph.getTensors().getOfType(TensorType::Variable);

  std::set<Tensor *> inverseLossScaleTensors;
  for (Tensor *variable : variables) {
    // find out if the tensor is connected to an optimizer
    auto isUpdatedVariable = [](Tensor *tensor) -> bool {
      bool isUpdated = false;
      for (Op *consumer : tensor->consumers.getOps()) {
        if (consumer->isConvertibleTo<VarUpdateOp>()) {
          isUpdated = true;
        }
      }
      return isUpdated;
    };
    // only variables updated by gradients can have lossScaling tensors
    // values like batchNorm's scale / bias tensors have a
    // VariableUpdateType::Copy, and don't have lossScaling tensor
    if (variable->getVariableUpdateType() != VariableUpdateType::Gradient) {
      continue;
    }
    // only variables connected to optimizers can have
    // lossScaling tensors, the detached variables don't have it
    if (!isUpdatedVariable(variable)) {
      continue;
    }
    if (ir.tensorExistsInInitialisers(variable->id)) {
      TensorId inverseLossScaleId =
          optimizer.getInverseLossScalingTensorId(*variable);
      if (graph.getTensors().contains(inverseLossScaleId)) {
        inverseLossScaleTensors.insert(
            graph.getTensors().get(inverseLossScaleId));
      } else {
        throw error("[AutomaticLossScale transform] Unable to find inverse "
                    "loss scale tensor, '{}', in graph '{}'",
                    inverseLossScaleId,
                    graph.id);
      }
    }
  }

  return inverseLossScaleTensors;
}

nonstd::optional<std::string> getEnvVar(const std::string &env_var) {
  char *res = std::getenv(env_var.c_str());
  if (res) {
    return std::string(res);
  } else {
    return nonstd::nullopt;
  }
}

nonstd::optional<std::string> getPopartEnvVar(const std::string &env_var) {
  return getEnvVar(logging::format("POPART_{}", env_var));
}

nonstd::optional<std::string> getPopXLEnvVar(const std::string &env_var) {
  return getEnvVar(logging::format("POPXL_{}", env_var));
}

std::ostream &operator<<(std::ostream &ss, const std::vector<std::size_t> &v) {
  appendSequence(ss, v);
  return ss;
}

void OpSearchHelper::pushConsumers(Tensor *t) {
  for (auto consumer : t->consumers.getOps()) {
    push(consumer);
  }
}

void OpSearchHelper::pushOutputConsumers(Op *op) {
  for (auto output : op->output->tensors()) {
    pushConsumers(output);
  }
}

void OpSearchHelper::pushInputProducers(Op *op) {
  for (auto input : op->input->tensors()) {
    if (input->hasProducer()) {
      push(input->getProducer());
    }
  }
}

int roundToInt(float d) { return static_cast<int>(std::roundf(d)); }

unsigned roundToUnsigned(float d) {
  return static_cast<unsigned>(std::roundf(d));
}

// convert a float to the DataType `dtype`
std::vector<char> convertFloatToDataType(DataType dtype, float data) {
  if (dtype == DataType::FLOAT) {
    return convertFloatTo<float>(data);
  }

  else if (dtype == DataType::FLOAT16) {
    return convertFloatTo<Half>(data);
  }

  else if (dtype == DataType::INT32) {
    return convertIntTo<int>(roundToInt(data));
  }

  else if (dtype == DataType::UINT32) {
    return convertUnsignedIntTo<uint32_t>(roundToUnsigned(data));
  }

  else if (dtype == DataType::INT16) {
    return convertIntTo<int16_t>(roundToInt(data));
  }

  else if (dtype == DataType::UINT16) {
    return convertUnsignedIntTo<uint16_t>(roundToUnsigned(data));
  }

  else if (dtype == DataType::INT8) {
    return convertIntTo<int8_t>(roundToInt(data));
  }

  else if (dtype == DataType::UINT8) {
    return convertUnsignedIntTo<uint8_t>(roundToUnsigned(data));
  }

  throw error("Can't convert float to DataType {}",
              getDataTypeInfoMap().at(dtype).name());
}

// convert a float to type T
template <typename T> std::vector<char> convertFloatTo(float data) {
  std::vector<char> data_out;
  T converted_data{data};
  data_out.resize(sizeof(T));
  *reinterpret_cast<T *>(data_out.data()) = converted_data;
  return data_out;
}

// convert an int to type T
template <typename T> std::vector<char> convertIntTo(int data) {
  std::vector<char> data_out;
  data_out.resize(sizeof(T));
  T converted_data{static_cast<T>(data)};
  *reinterpret_cast<T *>(data_out.data()) = converted_data;
  return data_out;
}

// convert an unsigned int to type T
template <typename T> std::vector<char> convertUnsignedIntTo(uint32_t data) {
  std::vector<char> data_out;
  T converted_data{static_cast<T>(data)};
  data_out.resize(sizeof(T));
  *reinterpret_cast<T *>(data_out.data()) = converted_data;
  return data_out;
}

// map negative indices to positive indices, and cast to uint64_t.
std::vector<uint64_t> getAxes_u64(const std::vector<int64_t> &axes,
                                  uint64_t outRank) {

  std::vector<uint64_t> axes_u64;
  for (auto d : axes) {
    if (d < 0) {
      d += outRank;
    }
    if (d < 0) {
      std::ostringstream oss;
      oss << "Invalid axis in getAxes_u64(axes=";
      poprithms::util::append(oss, axes);
      oss << ", outRank=" << outRank << "). ";
      throw error(oss.str());
    }
    d = d % outRank;
    axes_u64.push_back(d);
  }
  return axes_u64;
}

int64_t getReduceAxis(int64_t axis_, int64_t inShapeSize) {
  // Onnx 11 supports negative axis indexing for reduce.
  if (axis_ >= int64_t(0)) {
    return axis_;
  } else {
    return inShapeSize + axis_;
  }
}

void normalizeReduceAxes(std::vector<int64_t> &axes, int64_t inShapeSize) {
  for (int64_t i = 0; i < axes.size(); i++) {
    axes[i] = getReduceAxis(axes[i], inShapeSize);
  }
}

void validateReduceAxis(int64_t axis_,
                        int64_t inShapeSize,
                        const std::string &message) {

  if (inShapeSize == 0) {
    throw error("Reduce input rank must be greater than 0, invalid "
                "Reduce {}.",
                message);
  }

  // From the onnx spec:
  // Accepted range is [-r, r-1] where r = rank(data).
  if (axis_ > inShapeSize - 1 || axis_ < -inShapeSize) {
    throw error("Axis {} is out of acceptable range [{}, {}]",
                axis_,
                -inShapeSize,
                inShapeSize - 1);
  }
}

void validateReduceAxes(const std::vector<int64_t> &axes,
                        int64_t inShapeSize,
                        const std::string &message) {
  for (size_t i = 0; i < axes.size(); i++) {
    validateReduceAxis(axes[i], inShapeSize, message);
  }
}

namespace {
template <typename S, typename D>
void cast(const void *src, void *dst, size_t nelms) {
  for (size_t i = 0; i < nelms; ++i) {
    *(reinterpret_cast<D *>(dst) + i) =
        boost::numeric_cast<D, S>(*(reinterpret_cast<const S *>(src) + i));
  }
}
} // namespace

std::vector<char>
cast(DataType src, DataType dst, const void *data, size_t nbytes) {
  const DataTypeInfo *srcDataTypeInfo = &getDataTypeInfoMap().at(src);
  const DataTypeInfo *dstDataTypeInfo = &getDataTypeInfoMap().at(dst);

  size_t nelms     = nbytes / srcDataTypeInfo->nbytes();
  size_t dstnbytes = nelms * dstDataTypeInfo->nbytes();

  if (dstnbytes < nbytes) {
    logging::info("[cast] Narrowing cast from {} to {}", src, dst);
  }

  std::vector<char> outData(dstnbytes);

  const void *srcData = data;
  void *dstData       = static_cast<void *>(outData.data());

  auto err = [&src, &dst]() {
    throw error("[cast] Unsupported cast data types {} -> {}", src, dst);
  };

  try {
    switch (src) {
    case DataType::INT32:
      switch (dst) {
      case DataType::INT64:
        cast<int32_t, int64_t>(srcData, dstData, nelms);
        break;
      default:
        err();
      }
      break;
    case DataType::UINT32:
      switch (dst) {
      case DataType::UINT64:
        cast<uint32_t, uint64_t>(srcData, dstData, nelms);
        break;
      default:
        err();
      }
      break;
    case DataType::INT64:
      switch (dst) {
      case DataType::INT32:
        cast<int64_t, int32_t>(srcData, dstData, nelms);
        break;
      default:
        err();
      }
      break;
    case DataType::UINT64:
      switch (dst) {
      case DataType::UINT32:
        cast<uint64_t, uint32_t>(srcData, dstData, nelms);
        break;
      default:
        err();
      }
      break;
    default:
      err();
    }
  } catch (boost::bad_numeric_cast &e) {
    throw error("[cast] Cast {} -> {} failed: {}", src, dst, e.what());
  }

  return outData;
}

std::vector<char>
cast(DataType src, DataType dst, const std::vector<char> &data) {
  return cast(src, dst, static_cast<const void *>(data.data()), data.size());
}

TensorId getBaseTensorId(const TensorId &t) {
  int64_t i = t.size() - 1;
  if (!isdigit(t.at(i))) {
    return t;
  }
  while (i >= 0 && isdigit(t.at(i))) {
    i--;
  }
  if (i < 3 || t.at(i) != 't') {
    return t;
  }
  if (t.at(i - 1) != '_' || t.at(i - 2) != '_') {
    return t;
  }
  return t.substr(0, i - 2);
}

TensorId addScope(const Graph &g, const TensorId &t) {
  ParsedTensorId pTId(t, g.getIr());
  pTId.addScope(g.getScope());
  return pTId.getId();
}

TensorId removeScope(const Graph &g, const TensorId &t) {
  ParsedTensorId pTId(t, g.getIr());
  pTId.removeScope(g.getScope());
  return pTId.getId();
}

TensorId addPrefix(const Ir &ir, const TensorId &t, const TensorId &prefix) {
  ParsedTensorId pTId(t, ir);
  pTId.addPrefix(prefix);
  return pTId.getId();
}

TensorId removePrefix(const Ir &ir, const TensorId &t, const TensorId &prefix) {
  ParsedTensorId pTId(t, ir);
  pTId.removePrefixIfExist(prefix);
  return pTId.getId();
}

std::ostream &operator<<(std::ostream &ss,
                         const StochasticRoundingMethod &srm) {
  switch (srm) {
  case StochasticRoundingMethod::DifferingSeeds: {
    ss << "DifferingSeeds";
    break;
  }
  case StochasticRoundingMethod::IdenticalSeeds: {
    ss << "IdenticalSeeds";
    break;
  }
  default:
    throw error("Unsupported StochasticRoundingMethod value {}",
                static_cast<int>(srm));
  }

  return ss;
}

} // namespace popart
