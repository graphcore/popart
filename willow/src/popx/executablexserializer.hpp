// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_POPX_EXECUTABLEXSERIALIZER_HPP_
#define POPART_WILLOW_SRC_POPX_EXECUTABLEXSERIALIZER_HPP_

#include <iostream>
#include <memory>

#include <popef/Reader.hpp>

#include <popart/capnp/Ir.capnp.h>
#include <popart/commgroup.hpp>
#include <popart/tensor.hpp>
#include <popart/variablesettings.hpp>

#include "popart/datatype.hpp"

namespace popart {

// Forward declaration.
class Ir;

namespace popx {

// Forward declaration.
class Executablex;
class IrLowering;

namespace serialization {

/**
 * Helper function to map between representations of tensor types.
 */
popart::cap::TensorType toCapnpTensorType(popart::TensorType type);

/**
 * Helper function to map between representations of tensor types.
 */
popart::TensorType toPopartTensorType(popart::cap::TensorType type);

/**
 * Helper function to map between representations of tensor data types.
 */
popart::cap::DataType toCapnpDataType(popart::DataType type);

/**
 * Helper function to map between representations of tensor data types.
 */
popart::DataType toPopartDataType(popart::cap::DataType type);

/**
 * Helper function to map between representations of comm group types.
 */
popart::cap::CommGroupType toCapnpCommGroupType(popart::CommGroupType type);

/**
 * Helper function to map between representations of comm group types.
 */
popart::CommGroupType toPopartCommGroupType(popart::cap::CommGroupType type);

/**
 * Helper function to map between representations of variable retrieval modes.
 */
popart::cap::VariableRetrievalMode
toCapnpVariableRetrievalMode(popart::VariableRetrievalMode mode);

/**
 * Helper function to map between representations of variable retrieval modes.
 */
popart::VariableRetrievalMode
toPopartVariableRetrievalMode(popart::cap::VariableRetrievalMode mode);

/**
 * Serialise a tensor to capnp.
 * \param tensor The PopART tensor to serialise.
 * \param tensorBuilder The builder object to serialise to.
 */
void serializeTensor(const popart::Tensor *tensor,
                     popart::cap::Tensor::Builder &tensorBuilder);

/**
 * Deserialise a tensor from capnp.
 * \param ir The PopART IR to construct the tensor in.
 * \param capnpTensor The serialised capnp tensor.
 * \param tensorReader A pointer to an object which allows
 *                     getting serialized tensor data.
 * \return A unique pointer to the constructed tensor.
 */
std::unique_ptr<popart::Tensor>
deserializeTensor(popart::Ir &ir,
                  const popart::cap::Tensor::Reader &capnpTensor,
                  const popef::TensorReader *tensorReader);

/**
 * Serialise executable.
 * \param out The stream to serialise to.
 * \param executable The executable to serialise.
 */
void serializePopartExecutable(std::ostream &out,
                               const popart::popx::Executablex &executable);

/**
 * Deserialise executable.
 * \param in The stream to serialise from.
 * \param ir The IR to construct the executable with.
 * \param lowering The IR lowering object to construct the executable with.
 * \param tensorDataVec Vector of readable popef tensor data blobs. They contain
 *                      the serialized data for popart tensors.
 * \return A unique pointer to the constructed executable.
 */
std::unique_ptr<popart::popx::Executablex> deserializePopartExecutable(
    std::istream &in,
    popart::Ir &ir,
    popart::popx::IrLowering &lowering,
    const std::vector<popef::TensorReader> &tensorDataVec);

} // namespace serialization
} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_SRC_POPX_EXECUTABLEXSERIALIZER_HPP_
