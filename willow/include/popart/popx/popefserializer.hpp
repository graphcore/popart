// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_POPEFSERIALIZER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_POPEFSERIALIZER_HPP_

#include <cstddef>
#include <iostream>
#include <memory>

#include <popef/Types.hpp>
#include <popef/Writer.hpp>

namespace poplar {
// Forward declaration.
class Executable;
} // namespace poplar

namespace popart {

// Forward declaration.
class Ir;
class Tensor;
class TensorInfo;
enum class DataType;

namespace popx {

// Forward declaration.
class IrLowering;
class Executablex;
class Devicex;

namespace serialization {

// Forward declaration.
class ReaderImpl;

/**
 * The function casts \c popart::DataType to \c popef::DataType.
 *
 * \param type Tensor data type.
 * \return The created \c popef::DataType object.
 */
popef::DataType toPopefDataType(popart::DataType type);

/**
 * The function creates tensor info based on information contained
 * in passed arguments.
 *
 * \param dt The tensor data type.
 * \param shape The tensor shape.
 * \return The created \c popef::TensorInfo object.
 */
popef::TensorInfo createTensorInfo(const popef::DataType dt,
                                   const std::vector<int64_t> &shape);

/**
 * The function casts \c popart::TensorInfo to \c popef::TensorInfo.
 *
 * \param info Object that contains information about tensor.
 * \return The created \c popef::TensorInfo  object.
 */
popef::TensorInfo createTensorInfo(const popart::TensorInfo &info);

/**
 * The function serializes the content of the tensor to the popef
 * file (specified by the \c popef::Writer object) as a tensor data
 * blob. Such blob contains tensor name, its shape, and data type
 * with binary data.
 *
 * \param tensor Popart tensor.
 * \param tensorInfo Object that contains information about the tensor.
 * \param writer Write popef tensor data blob to a given stream.
 */
void serializePopefTensor(const popart::Tensor &tensor,
                          const popef::TensorInfo &tensorInfo,
                          popef::Writer &writer);

/**
 * The function prepares popef file which can be used to run a model
 * using model_runtime or serving services supported by Graphcore.
 * In detail: It serializes both the poplar engine's executable,
 * popart executable, the hash to the given ostream, non-user input
 * tensors and prepares popef metadata.
 *
 * \param out Destination stream to which data will be serialized.
 * \param device Devicex class has all data that are needed to
 *               execute proper serialization process.
 */
void serializeEngineExecutable(std::ostream &out,
                               const popart::popx::Devicex &device);

/**
 * \class Reader
 * \brief Reader is a class which facilitates deserialization process.
 * The most important advantage is the execution reading popef
 * file process once.
 */
class Reader {
public:
  /**
   * Constructs Reader class object.
   *
   * \param in Source stream from which a popef file will be read.
   */
  Reader(std::shared_ptr<std::istream> in);

  /**
   * Move constructor.
   */
  Reader(Reader &&reader);

  /**
   * Default destructor.
   */
  ~Reader();

  /**
   * \return The executable hash or 0 if the stream contains
   *         corrupted data.
   */
  size_t readExecutableHash() const;

  /**
   * \return True if the stream contains a Poplar executable.
   */
  bool containsPoplarExecutable() const;

  /**
   * \return True if the stream contains a Popart executable.
   */
  bool containsExecutable() const;

  /**
   * \return True if the stream contains a Popef metadata.
   */
  bool containsPopefMetadata();

  /**
   * Deserializes poplar executable from executable blob which
   * is part of a popef file.
   *
   * \return Poplar executable.
   */
  poplar::Executable deserializePoplarExecutable() const;

  /**
   * Load a popart executable from a popef file.
   *
   * \param ir Object of \c popart::Ir class to which some of the
   *         deserialized data will be write.
   * \param lowering Object of \c popart::popx::IrLowering class to which
   *        some of the deserialized data will be write.
   * \return Popart executable.
   */
  std::unique_ptr<popart::popx::Executablex>
  deserializeExecutable(popart::Ir &ir,
                        popart::popx::IrLowering &lowering) const;

private:
  std::unique_ptr<ReaderImpl> _impl;
};

} // namespace serialization
} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_POPEFSERIALIZER_HPP_
