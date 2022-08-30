// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_POPX_POPEFSERIALIZERIMPL_HPP_
#define POPART_WILLOW_SRC_POPX_POPEFSERIALIZERIMPL_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include <popef/Reader.hpp>
#include <popef/Types.hpp>
#include <popef/Writer.hpp>

#include <popart/vendored/optional.hpp>

namespace poplar {
// Forward declaration.
class Executable;
class Engine;
class OptionFlags;
} // namespace poplar

namespace popart {

// Forward declaration.
class DeviceInfo;
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

class Reader;

/**
 * Cast \c popart::DataType to <tt>popef::DataType</tt>.
 *
 * \param type Tensor data type.
 *
 * \return The created \c popef::DataType object.
 */
popef::DataType toPopefDataType(DataType type);

/**
 * Create tensor information based on the arguments.
 *
 * \param dt The tensor data type.
 * \param shape The tensor shape.
 *
 * \return The created \c popef::TensorInfo object.
 */
popef::TensorInfo createTensorInfo(const popef::DataType dt,
                                   const std::vector<int64_t> &shape);

/**
 * Cast \c popart::TensorInfo to \c popef::TensorInfo.
 *
 * \param info Information about the tensor.
 *
 * \return The created \c popef::TensorInfo  object.
 */
popef::TensorInfo createTensorInfo(const popart::TensorInfo &info);

/** Implementation of Writer class. */
class WriterImpl {
private:
  // Mapping replica id to the rng state buffer.
  using RngStateBufferType = std::map<uint16_t, std::vector<uint32_t>>;

public:
  /**
   * Constructs WriterImpl class object.
   *
   * \param out Destination stream to which data will be serialized.
   * \param device All the data that is needed to serialize PopART state.
   */
  WriterImpl(std::ostream &out, const popart::popx::Devicex &device);

  /**
   * Serialize the content of the tensor to the PopEF
   * file (specified by the \c popef::Writer object) as a tensor data
   * blob. This blob contains the tensor name, shape, and data type
   * with binary data.
   *
   * \param tensor Tensor to be serialized.
   * \param tensorInfo Information about the tensor.
   * \param writer Will write PopEF tensor data blob to a stream.
   */
  static void serializePopefTensor(const popart::Tensor &tensor,
                                   const popef::TensorInfo &tensorInfo,
                                   popef::Writer &writer);

  /**
   * Export the model's tensors content. Data mainly comes from the executablex
   * object with one minor exception: rngBuffer which comes from devicex
   * object.
   */
  void serializeTensorData();

  /** Export PopART-specific data. */
  void serializePopartMetadata();

  /** Export Poplar engine's executable. */
  void serializePoplarEngine();

  /**
   * Serializes the data needed to run a Poplar executable (using the Model
   * runtime library) created by PopART
   */
  void serializePopefMetadata();

private:
  static constexpr const char *rngStateTensorName = "rngStateTensor";

  /**
   * Copy programs that users can run on the IPU using
   * the compiled Poplar executable.
   */
  const std::unordered_map<popef::ProgramFlow::ProgramIndexType, std::string>
  createProgramsMap() const;

  /**
   * Create PopEF Anchor object from the arguments.
   *
   * \param name Tensor name.
   * \param handle A string key that connects the callback with the tensor.
   * \param tensorInfo Information about the tensor.
   * \param isPerReplica Whether this tensor should have separate
   *                     data for each replica.
   * \param type Whether this is an input or output tensor.
   * \param programs A list of programs where the tensor is used.
   */
  static popef::Anchor createAnchor(
      const std::string &name,
      const std::string &handle,
      const popef::TensorInfo &tensorInfo,
      const bool isPerReplica,
      const popef::TensorType type,
      const std::vector<popef::ProgramFlow::ProgramIndexType> &programs);

  /**
   * \brief Create PopEF Anchor object from the arguments.
   *
   * The anchor is inserted into a vector of anchors which can be serialized
   * as a tensor data blob to the PopEF file.
   * The tensor type is set to input.
   *
   * \param name Tensor name.
   * \param handle A string key that connects the callback with the tensor.
   * \param tensorInfo Information about the tensor.
   * \param isPerReplica Whether this tensor should have separate
   *                     data for each replica.
   * \param programs A list of programs where the tensor is used.
   */
  void addDataInputAnchor(
      const std::string &name,
      const std::string &handle,
      const popef::TensorInfo &tensorInfo,
      const bool isPerReplica,
      const std::vector<popef::ProgramFlow::ProgramIndexType> &programs);

  /**
   * \brief Create PopEF Anchor object from the arguments.
   *
   * The anchor is inserted into a vector of anchors which cannot be serialized
   * as a tensor data blob to the PopEF file.
   * The tensor type is set to input.
   *
   * \param name Tensor name.
   * \param handle A string key that connects the callback with the tensor.
   * \param tensorInfo Information about the tensor.
   * \param isPerReplica Whether this tensor should have separate
   *                     data for each replica.
   * \param programs A list of programs where the tensor is used.
   */
  void addInputAnchor(
      const std::string &name,
      const std::string &handle,
      const popef::TensorInfo &tensorInfo,
      const bool isPerReplica,
      const std::vector<popef::ProgramFlow::ProgramIndexType> &programs);

  /**
   * \brief Create PopEF Anchor object from the arguments.
   *
   * The anchor is inserted into a vector of anchors which cannot be serialized
   * as a tensor data blob to the PopEF file.
   * The tensor type is set to output.
   *
   * \param name Tensor name.
   * \param handle A string key that connects the callback with the tensor.
   * \param tensorInfo Information about the tensor.
   * \param isPerReplica Whether this tensor should have separate
   *                     data for each replica.
   * \param programs A list of programs where the tensor is used.
   */
  void addOutputAnchor(
      const std::string &name,
      const std::string &handle,
      const popef::TensorInfo &tensorInfo,
      const bool isPerReplica,
      const std::vector<popef::ProgramFlow::ProgramIndexType> &programs);

  /**
   * \brief Create PopEF Anchor object from the arguments.
   *
   * The anchor is inserted into a vector of anchors which can be serialized
   * as a tensor data blob to the PopEF file.
   * The tensor type is set to unknown. It is needed only to restore
   * PopART state during deserialization process.
   *
   * \param name Tensor name.
   * \param tensorInfo Information about the tensor.
   * \param isPerReplica Whether this tensor should have separate
   *                     data for each replica.
   */
  void addDataUnknownAnchor(const std::string &name,
                            const popef::TensorInfo &tensorInfo,
                            bool isPerReplica);

  /**
   * Create objects (based on model weights) of
   * \c popef::Anchor type and copies them to the appropriate container.
   * The container choice depends on whether the tensor has tensor data
   * to save.
   */
  void createPopefAnchorsFromWeights();

  /**
   * Create objects (based on optimizer tensors) of
   * \c popef::Anchor type and copies them to the appropriate container.
   * The container choice depends on whether the tensor has tensor data
   * to save.
   */
  void createPopefAnchorsFromOptimizers();

  /**
   * Create objects (based on data stream tensors) of
   * \c popef::Anchor type and copies them to the appropriate container.
   * The container choice depends on whether the tensor has tensor data
   * to save.
   */
  void createPopefAnchorsFromDataStreams();

  /**
   * Create objects (based on anchor tensors) of
   * \c popef::Anchor type and copies them to the appropriate container.
   * The container choice depends on whether the tensor has tensor data
   * to save.
   */
  void createPopefAnchorsFromAnchors();

  /**
   * Create objects (based on random seed tensor) of
   * \c popef::Anchor type and copies them to the appropriate container.
   * The container choice depends on whether the tensor has tensor data
   * to save.
   */
  void createPopefAnchorsFromRandomSeed();

  /**
   * Create objects (based on rng state tensors) of
   * \c popef::Anchor type and copies them to the appropriate container.
   * The container choice depends on whether the tensor has tensor data
   * to save.
   */
  void createPopefAnchorsFromRNGState();

  /**
   * Create objects (based on cycle counter tensors) of
   * \c popef::Anchor type and copies them to the appropriate container.
   * The container choice depends on whether the tensor has tensor data
   * to save.
   */
  void createPopefAnchorsFromCycleCounters();

  /** Create anchors based on PopART tensors. */
  void createAnchors();

  /** Export content of RNG state tensors. */
  void serializeRngBufferContent(const popef::Anchor &rngBufferAnchor);

  /**
   * \param deviceInfo Represents a target device of the model.
   *
   * \return The IPU version that has been declared by the user for compilation
   *         the model.
   */
  static int getIpuVersion(const DeviceInfo &deviceInfo);

  /**
   * \param deviceInfo Represents a target device of the model.
   *
   * \return Whether the target runtime system is a Pod.
   */
  static bool isPOD(const DeviceInfo &deviceInfo);

  /**
   * \param optFlags A set of option and value
   *        string flags.
   *
   * \return vector of \c popef::Option casted from vector of
   *         \c poplar::OptionFlags.
   *
   * \note Available options:
   *       - <a
   *         href="https://docs.graphcore.ai/projects/poplar-api/en/latest/using_libs.html#option-values">engine
   *         options</a>
   *       - device options: The configuration for the created device. See
   *                         createCpuDevice(), createIpuModelDevice(),
   *                         createOfflineIPUDevice() and createSimDevice()
   *                         for more information about options.
   */
  static std::vector<popef::Option>
  convertOptionFlagsToOptions(const poplar::OptionFlags &optFlags);

  // Write PopEF blobs to a given stream.
  popef::Writer _writer;
  // The final executable which contains all the data, metadata
  // and configuration parameters necessary to start running
  // the program on the device.
  const popart::popx::Executablex &_executablex;
  // The map with all popart programs:
  //   - the key: program id
  //   - the value: program name
  std::unordered_map<popef::ProgramFlow::ProgramIndexType, std::string>
      popartPrograms;
  // The Engine class provides the ability to execute a graph program.
  // It also gives a possibility to serialize poplar executable.
  const std::unique_ptr<poplar::Engine> &_engine;
  std::string _programHash;
  // Content of rng state buffer for all replicas.
  const RngStateBufferType &_rngBuffer;
  // Seed tensor's callback handle.
  std::string _seedHandle;
  // Anchors which need data serialization.
  std::vector<popef::Anchor> _anchorsWithData;
  // Anchors which do not need data serialization.
  std::vector<popef::Anchor> _anchorsWithoutData;
};

/** Implementation of Reader class. */
class ReaderImpl {
public:
  /**
   * Constructs ReaderImpl class object.
   *
   * \param in Vector of source streams from which a PopEF file
   *           will be read.
   */
  ReaderImpl(const std::vector<std::shared_ptr<std::istream>> &in_vec);

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
   * \param ir Object which some of the deserialized data
   *        will be written to.
   * \param lowering Object which some of the deserialized
   *        data will be written to.
   *
   * \return PopART executable.
   */
  std::unique_ptr<popart::popx::Executablex>
  deserializeExecutable(popart::Ir &ir,
                        popart::popx::IrLowering &lowering) const;

private:
  friend class Reader;

  template <typename T>
  using optional_ref    = nonstd::optional<std::reference_wrapper<T>>;
  using OpaqueReaderOpt = optional_ref<const popef::OpaqueReader>;
  using ExecReaderOpt   = optional_ref<const popef::ExecutableReader>;
  using MetadataOpt     = optional_ref<const popef::Metadata>;
  using OpaqueReaderIt  = std::vector<popef::OpaqueReader>::const_iterator;
  using ExecReaderIt    = std::vector<popef::ExecutableReader>::const_iterator;
  using MetadataIt      = std::vector<popef::Metadata>::const_iterator;

  /**
   * Create a \c popef::Reader object and parse a stream that contains
   * PopEF file.
   *
   * \param in Stream that contains PopEF file to be parsed.
   *
   * \return Object containing parsed PopEF stream.
   */
  static popef::Reader
  setupReader(const std::vector<std::shared_ptr<std::istream>> &in_vec);

  /**
   * \return Opaque blob which contains serialized PopART-specific data.
   */
  OpaqueReaderOpt findPopartOpaque();

  /**
   * \return Poplar executable blob.
   */
  ExecReaderOpt findPoplarExecutable();

  /**
   * \return PopEF metadata.
   */
  MetadataOpt findPopefMetadata();

  /**
   * \return All tensor blobs associated with anchors from the PopEF metadata.
   */
  std::vector<popef::TensorReader> findPopefTensors();

  /**
   * \return The executable hash or 0 if the stream contains
   *         corrupted data.
   */
  size_t getExecutableHash() const;

  popef::Reader _popefReader;
  const OpaqueReaderOpt _popartOpaque;
  const ExecReaderOpt _poplarExecutable;
  const MetadataOpt _popefMetadata;
  const std::vector<popef::TensorReader> _tensorDataVec;
  const size_t _hash;
};

} // namespace serialization
} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_SRC_POPX_POPEFSERIALIZERIMPL_HPP_
