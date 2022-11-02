// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_UTIL_FLOAT8_HPP_
#define POPART_WILLOW_INCLUDE_POPART_UTIL_FLOAT8_HPP_

#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include <string>
#include <poplar/DebugContext.hpp>
#include <poplar/Quarter.hpp>

namespace poplar {
class Tensor;
class Graph;
namespace program {
class Sequence;
}
} // namespace poplar

namespace popart {
class TensorIndexMap;

/**
 * @brief Returns true iff the type is a float8 type.
 *
 * @param type IR datatype
 */
inline bool isFloat8(DataType type) {
  return type == DataType::FLOAT8_143 || type == DataType::FLOAT8_152;
}

/**
 * @brief Raise an error if the Op inputs contain an invalid combination of
 * float8 inputs and do nothing otherwise. For example, if the op mixes float16
 * with float8 operands, this function will throw.
 *
 * @param inputs Op inputs tensors
 * @param log2ScaleTensorIdx The index where the log2scale tensor is located in
 * the inputs
 * @param debugName the debug name of the op to be included in error messages.
 */
void validateOpFloat8Inputs(const TensorIndexMap *inputs,
                            const InIndex log2ScaleTensorIdx,
                            std::string debugName);

/**
 * @brief Reinterpret an UNSIGNED_CHAR tensor containing FLOAT8_{152|143} data
 * as QUARTER.
 *
 * @param g The poplar graph
 * @param x Tensor of type unsigned char containing float8 data
 * @param metadata Tensor of type QUARTER_METADATA
 * @param prog The poplar program.
 * @param dc debug context.
 * @return poplar::Tensor
 */
poplar::Tensor reinterpretCastUInt8ToQuarter(poplar::Graph &g,
                                             poplar::Tensor &x,
                                             poplar::Tensor &metadata,
                                             poplar::program::Sequence &prog,
                                             poplar::DebugContext dc = {});

/**
 * @brief Reinterpret an UNSIGNED_CHAR tensor containing FLOAT8_{152|143} data
 * as QUARTER.
 *
 * @param graph The poplar graph.
 * @param x Tensor of type UNSIGNED_CHAR containing float8 data.
 * @param format the float8 format. Can be one of F143 or F152
 * @param log2Scale a scalar poplar tensor of signed integral type containing
 * the scale bias.
 * @param prog poplar sequence to add the reinterpret-cast to.
 * @param dc debug context.
 * @return poplar::Tensor of type QUARTER with metadata containing the value of
 * `log2scale`
 */
poplar::Tensor
reinterpretCastUInt8ToQuarter(poplar::Graph &graph,
                              poplar::Tensor &x,
                              poplar::QuarterMetadata::Format format,
                              poplar::Tensor &log2Scale,
                              poplar::program::Sequence &prog,
                              poplar::DebugContext dc = {});

/**
 * @brief Reinterpret an UNSIGNED_CHAR tensor containing FLOAT8_{152|143} data
 * as QUARTER.
 *
 * @param graph The poplar graph.
 * @param x Tensor of type UNSIGNED_CHAR containing float8 data.
 * @param format the PopART IR format.
 * @param log2Scale a signed integer type containing
 * the scale bias.
 * @param prog poplar sequence to add the reinterpret-cast to.
 * @param dc debug context.
 * @return poplar::Tensor of type QUARTER with metadata containing the value of
 * `log2scale`
 */
poplar::Tensor
reinterpretCastUInt8ToQuarter(poplar::Graph &graph,
                              poplar::Tensor &x,
                              poplar::QuarterMetadata::Format format,
                              int log2Scale,
                              poplar::program::Sequence &prog,
                              poplar::DebugContext dc = {});

/**
 * @brief Create a poplar program that throws an error if the log2 scale tensor
 * is not in the range [lower, upper), and does nothing otherwise.
 *
 *  The program is semantically equivalent to:
 *  if (log2scale >= upper || log2scale < lower) {
 *   throw error
 *  }
 *
 * This function is used in some fused ops that support a log2 scale tensor
 * argument to perform a runtime check on the value in the tensor.
 * @param graph The poplar graph.
 * @param log2Scale A scalar tensor of signed integral type.
 * @param lower integer lower bound, inclusive.
 * @param upper integer upper bound, exclusive.
 * @param dc debug context.
 * @return poplar::program::Sequence a
 */
poplar::program::Sequence
createAssertLog2ScaleInRangeProg(poplar::Graph &graph,
                                 poplar::Tensor &log2Scale,
                                 int lower,
                                 int upper,
                                 poplar::DebugContext dc = {});

/**
 * @brief Convert a PopART FLOAT8_{152|143} DataType format to its
 * equivalent Poplar QuarterMetadata format.
 *
 * @param type either DataType::FLOAT8_143 or DataType::FLOAT8_152
 * @return poplar::QuarterMetadata::Format
 */
poplar::QuarterMetadata::Format toPoplarQuarterFormat(DataType type);

/**
 * @brief Returns true iff the op inputs given by `inputs` are a valid
 * set of tensors inputs for a power of 2 scaled op variant.
 *
 * @param inputs
 * @param log2ScaleTensorIdx
 * @return true
 * @return false
 */
bool opInputsAreValidPow2ScaledInputs(const TensorIndexMap *inputs,
                                      const InIndex log2ScaleTensorIdx);
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_UTIL_FLOAT8_HPP_
