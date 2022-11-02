// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "popart/util/float8util.hpp"
#include "popart/datatype.hpp"
#include "popart/tensor.hpp"
#include "popart/tensorindex.hpp"
#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>
#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/MetadataCreation.hpp>
#include <poplar/Program.hpp>
#include <poplar/Quarter.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

namespace pe = popops::expr;

namespace popart {

std::set<DataType> inputTensorTypes(const TensorIndexMap *inputs,
                                    const InIndex log2ScaleTensorIdx) {
  std::set<DataType> types;
  for (auto [id, t] : inputs->tensorMap()) {
    if (id != log2ScaleTensorIdx) {
      types.insert(t->info.dataType());
    }
  }
  return types;
}

bool allInputsAreFloat8(const std::set<DataType> &types) {
  return std::all_of(types.begin(), types.end(), isFloat8);
}

void validateOpFloat8Inputs(const TensorIndexMap *inputs,
                            const InIndex log2ScaleTensorIdx,
                            std::string debugName) {
  auto types     = inputTensorTypes(inputs, log2ScaleTensorIdx);
  bool allFloat8 = allInputsAreFloat8(types);

  bool someAreFloat8 = std::any_of(types.begin(), types.end(), isFloat8);

  if (!allFloat8 && someAreFloat8) {
    // we know there is at least one type was a float8
    auto it = std::find_if_not(types.begin(), types.end(), isFloat8);
    throw error("Invalid operand type: {} in op {}.  If using a "
                "FLOAT8 input, all op operands must be a FLOAT8 type",
                *it,
                debugName);
  }

  // check if all float8 then log2scale must be present
  bool hasLog2Scale = inputs->hasIndex(log2ScaleTensorIdx);
  if (allFloat8 && !hasLog2Scale) {
    throw error("Log2 scale input tensor must be provided for FLOAT8 input "
                "types in op {}",
                debugName);
  }

  if (!allFloat8 && hasLog2Scale) {
    throw error("Log2 scale input tensor not accepted for non-FLOAT8 input "
                "types in op {}",
                debugName);
  }

  // if log2scale present then it must have correct type and shape
  if (hasLog2Scale) {
    const Tensor *log2ScaleTensor = inputs->tensor(log2ScaleTensorIdx);
    if (auto type = log2ScaleTensor->info.dataType(); type != DataType::INT32) {
      throw error(logging::format("Invalid log2 scale input type {} in op {}."
                                  "Log2 scale input tensor must "
                                  "be of type INT32. ",
                                  type,
                                  debugName));
    }

    if (int rank = log2ScaleTensor->info.rank(); rank > 0) {
      throw error(logging::format("Log2 scale must be a scalar tensor. "
                                  "Provided tensor has rank {} in op {}",
                                  rank,
                                  debugName));
    }
  }
}

poplar::Tensor reinterpretCastUInt8ToQuarter(poplar::Graph &g,
                                             poplar::Tensor &x,
                                             poplar::Tensor &metadata,
                                             poplar::program::Sequence &prog,
                                             poplar::DebugContext dc) {

  if (auto type = x.elementType(); type != poplar::UNSIGNED_CHAR) {
    throw error("Invalid tensor type for conversion to quarter: {}", type);
  }

  // We can't directly reinterpret UINT8 tensors as QUARTER, so we have
  // to workaround using copy-cloning. Poplar elides this.
  auto float8Tensor = g.clone(poplar::QUARTER, metadata, x);
  prog.add(poplar::program::Copy(
      x, float8Tensor.reinterpret(poplar::UNSIGNED_CHAR)));

  return float8Tensor;
}

poplar::Tensor
reinterpretCastUInt8ToQuarter(poplar::Graph &graph,
                              poplar::Tensor &x,
                              poplar::QuarterMetadata::Format format,
                              poplar::Tensor &log2Scale,
                              poplar::program::Sequence &prog,
                              poplar::DebugContext dc) {

  if (auto type = x.elementType(); type != poplar::UNSIGNED_CHAR) {
    throw error("Invalid tensor type for conversion to quarter: {}", type);
  }

  auto metadata =
      poplar::createVariableMetadataTensor(graph, format, log2Scale, prog, dc);

  return reinterpretCastUInt8ToQuarter(graph, x, metadata, prog, dc);
}

poplar::Tensor
reinterpretCastUInt8ToQuarter(poplar::Graph &graph,
                              poplar::Tensor &x,
                              poplar::QuarterMetadata::Format format,
                              int log2Scale,
                              poplar::program::Sequence &prog,
                              poplar::DebugContext dc) {

  auto metadata =
      poplar::createVariableMetadataTensor(graph, format, log2Scale, dc);

  return reinterpretCastUInt8ToQuarter(graph, x, metadata, prog, dc);
}

poplar::program::Sequence
createAssertLog2ScaleInRangeProg(poplar::Graph &graph,
                                 poplar::Tensor &log2Scale,
                                 int lower,
                                 int upper,
                                 poplar::DebugContext dc) {
  poplar::program::Sequence seq;

  // x >= upper or x < lower
  auto outsideRange = pe::Or(pe::Gte(pe::_1, pe::Const(upper)),
                             pe::Lt(pe::_1, pe::Const(lower)));

  auto errorBranch = poplar::program::ErrorProgram(
      logging::format(
          "Log2 scale is not in the range [{}, {}). ", lower, upper),
      log2Scale,
      dc);

  auto notInRange = popops::map(graph, outsideRange, {log2Scale}, seq);

  seq.add(poplar::program::If(
      notInRange, errorBranch, poplar::program::Sequence(), dc));

  return seq;
}

poplar::QuarterMetadata::Format toPoplarQuarterFormat(DataType type) {
  switch (type) {
  case popart::DataType::FLOAT8_143:
    return poplar::QuarterMetadata::Format::F143;
  case popart::DataType::FLOAT8_152:
    return poplar::QuarterMetadata::Format::F152;
  default:
    throw error("Invalid FLOAT8 datatype: {}", type);
  }
}

bool opInputsAreValidPow2ScaledInputs(const TensorIndexMap *inputs,
                                      const InIndex log2ScaleTensorIdx) {
  if (!inputs->hasIndex(log2ScaleTensorIdx)) {
    return false;
  }
  auto types                   = inputTensorTypes(inputs, log2ScaleTensorIdx);
  bool allFloat8               = allInputsAreFloat8(types);
  auto log2ScaleInfo           = inputs->tensor(log2ScaleTensorIdx)->info;
  bool log2ScaleHasCorrectType = log2ScaleInfo.dataType() == DataType::INT32;
  bool log2ScaleIsScalar       = log2ScaleInfo.rank() == 0;
  return allFloat8 && log2ScaleHasCorrectType && log2ScaleIsScalar;
}
} // namespace popart
