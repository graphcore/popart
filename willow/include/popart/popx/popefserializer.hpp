// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_POPEFSERIALIZER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_POPEFSERIALIZER_HPP_

#include <cstddef>
#include <iostream>
#include <memory>

namespace poplar {
// Forward declaration.
class Executable;
} // namespace poplar

namespace popart {

// Forward declaration.
class Ir;

namespace popx {

// Forward declaration.
class IrLowering;
class Executablex;
class Devicex;

namespace serialization {

// Forward declaration.
class ReaderImpl;

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
