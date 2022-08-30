// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_POPEFSERIALIZER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_POPEFSERIALIZER_HPP_

#include <iostream>
#include <memory>
#include <vector>

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
class WriterImpl;

/**
 * Save a compiled graph with additional data to a stream.
 *
 * PopART is able to save its state after the model compilation is complete,
 * so that it can be restored at a later time. To make this possible,
 * it is necessary to save such elements as:
 *   - a serialised Poplar executable,
 *   - its associated metadata,
 *   - tensor data blobs if model parameters have not been frozen
 *     (refer to the \c SessionOptions::constantWeights for more
 *     information),
 *   - a PopART-specific opaque blob to store information only
 *     relevant to PopART. This is needed to restore PopART state.
 *
 * All of these is possible using Writer class. The class will write <a
 * href="https://docs.graphcore.ai/projects/popef/en/latest/">PopEF</a>
 * blobs with data mentioned above to the stream provided in the constructor.
 * This can be used to restore PopART state or run a model using the
 * Model runtime or serving services supported by Graphcore.
 */
class Writer {
public:
  /**
   * Constructs Writer class object.
   *
   * \param out Destination stream to which data will be serialized.
   * \param device All the data that is needed to serialize PopART state.
   */
  Writer(std::ostream &out, const popart::popx::Devicex &device);

  /**
   * Move constructor.
   */
  Writer(Writer &&reader);

  /**
   * Default destructor.
   */
  ~Writer();

  /**
   * Serializes the Poplar engine executable and metadata
   * needed to run it outside the PopART environment.
   */
  void serializePoplarExecutable();

  /**
   * Serializes the PopART opaque blob. These are the data
   * needed to restore the PopART state.
   */
  void serializePopartMetadata();

  /**
   * Serializes the data for the following group of tensors:
   *   - weights,
   *   - optimizers,
   *   - random seed,
   *   - RNG state tensors.
   */
  void serializeTensorData();

private:
  std::unique_ptr<WriterImpl> _impl;
};

/**
 * \brief A class which facilitates deserialization process.
 *
 * It allows reading serialized streams allowing restoring PopART state.
 * For more information on what components are deserialized please refer
 * to \c Writer class.
 */
class Reader {
public:
  /**
   * Constructs Reader class object.
   *
   * \param in Vector of source streams from which a PopEF file
   *           will be read.
   */
  Reader(const std::vector<std::shared_ptr<std::istream>> &in_vec);

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
   * \return True if the stream contains a PopEF metadata.
   */
  bool containsPopefMetadata();

  /**
   * Deserializes Poplar executable from an executable blob which
   * is part of a PopEF file.
   *
   * \return Poplar executable.
   */
  poplar::Executable deserializePoplarExecutable() const;

  /**
   * Load a PopART executable from a PopEF file.
   *
   * \param ir Object which some of the deserialized data will
   *        be written to.
   * \param lowering Object  which some of the deserialized
   *        data will be written to.
   *
   * \return PopART executable.
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
