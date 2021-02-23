// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/popx/exporter.hpp>

#include <poplar/Engine.hpp>
#include <poplar/OptionFlags.hpp>
#include <popart/builder.hpp>
#include <popart/error.hpp>
#include <popart/istepio.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/popx/devicex.hpp>

#ifdef POPLAR_RUNNER
#include <ipu/poplar_executable_data.h>
#endif

namespace popart {
namespace popx {
namespace {

#ifndef POPLAR_RUNNER
const std::string errorMsg =
    "Trying to {} from a PopART build that doesn't support it.\nMake sure to "
    "pass '-DPoplarRunner_INSTALL_DIR=/path/to/libpoplar_executable_runner/' "
    "to CMake";

#else  // POPLAR_RUNNER

void setIpuShape(ipu::TensorInfo &ipuInfo, const TensorInfo &info) {
  ipu::DataType type;
  switch (info.dataType()) {
  case DataType::INT32:
    type = ipu::DataType::S32;
    break;
  case DataType::FLOAT:
    type = ipu::DataType::F32;
    break;
  case DataType::FLOAT16:
    type = ipu::DataType::F16;
    break;
  default:
    throw error("Unsupported tensor DataType {}",
                getDataTypeInfoMap().at(info.dataType()).name());
  }
  ipuInfo.SetShape(ipu::TensorShape(info.shape(), type));
}

void validateInfeedInfo(ipu::Metadata &meta,
                        const std::string &feedName,
                        const ipu::TensorInfo &info) {
  for (const auto &infeed : meta.infeeds) {
    if (infeed.name == feedName) {
      if (infeed.streams.size() != 1) {
        throw error("All Popart feeds are made of exactly one stream");
      }
      if (!infeed.streams.front().TypeAndShapeMatch(info)) {
        std::stringstream ss;
        ss << "Popart info " << info.ToString()
           << " doesn't match the info from the metadata "
           << infeed.streams.front().ToString() << " for the feed named "
           << feedName;
        throw error("{}", ss.str());
      }
      // Feed found and type + shape match.
      return;
    }
  }
  throw error("Couldn't find a feed named " + feedName + " in the metadata");
}

void exportFeedContent(ipu::FeedWriter &writer,
                       const TensorInfo &feedInfo,
                       int64_t numElements,
                       IStepIO &step,
                       const std::string &feedName) {
  std::vector<int32_t> conversionBuffer;
  for (int64_t n = 0; n < numElements; n++) {
    if (n % 1000 == 0) {
      logging::debug("Exporting {}/{} from {}", n, numElements, feedName);
    }
    ConstVoidData data = step.in(feedName, feedInfo.nelms(), false);

    // check the shape
    if (data.info.shape() != feedInfo.shape()) {
      std::stringstream ss;
      ss << "The shape provided " << data.info.shape()
         << " didn't match the one expected " << feedInfo.shape()
         << " for the input " << feedName << " (element " << n << ")";
      throw error(ss.str());
    }

    // check the type
    if (data.info.dataType() == feedInfo.dataType()) {
      writer.AppendTensor(data.data);
    } else if (data.info.dataType() == DataType::INT64 &&
               feedInfo.dataType() == DataType::INT32) {

      static bool loggingWarning = false;
      if (loggingWarning == false) {
        logging::devicex::warn("Copying (host) tensor {} from INT64 to "
                               "INT32. Will only warn once",
                               feedName);
        loggingWarning = true;
      }
      if (conversionBuffer.empty()) {
        conversionBuffer.resize(feedInfo.nelms());
      }
      int32_t *dest      = conversionBuffer.data();
      const int64_t *src = static_cast<const int64_t *>(data.data);
      for (int i = 0; i < feedInfo.nelms(); ++i) {
        dest[i] = static_cast<int32_t>(src[i]);
      }
      writer.AppendTensor(conversionBuffer.data());
    } else {
      std::stringstream ss;
      ss << "Type discrepency for tensor " << feedName
         << ". User provided : " << data.info.data_type()
         << " and expected : " << feedInfo.data_type()
         << ". Consider a custom copy here (as memcpy cannot be used)";
      throw error(ss.str());
    }
    step.inComplete(feedName, feedInfo.nelms());
  }
  logging::info("Successfully exported {}/{} from {}",
                numElements,
                numElements,
                feedName);
}

ipu::MetadataBuilder
createBuilderAndExportWeights(const Devicex &device,
                              const std::string *weightsPath = nullptr) {

  ipu::MetadataBuilder builder;
  std::unique_ptr<ipu::BinaryWriter> weights_writer;
  if (weightsPath) {
    const std::string filename =
        (weightsPath->empty() ? "" : *weightsPath + "/") + "weights.bin";
    weights_writer = std::make_unique<ipu::BinaryWriter>(filename);
  }
  // Listing input / output parameters
  for (auto id : device.ir().getTensorIds(TensorType::Variable)) {
    Tensor *tensor = device.ir().getTensor(id);
    if (!tensor->hasProducer() &&
        !device.ir().streamingIsDisabledForTensor(id)) {
      ipu::TensorInfo info;
      info.SetHandle(device.h2dId(tensor->id));
      info.SetName(tensor->id);
      setIpuShape(info, tensor->info);
      builder.AddInputParameter(info);

      if (weights_writer) {
        if (tensor->info.nbytes() != info.Shape().DataSizeInBytes()) {
          throw internal_error(
              "Size mismatch between metadata and data for parameter {}",
              tensor->id);
        }
        info.SetType(ipu::TensorType::Parameter);
        ipu::Tensor out{info, tensor->tensorData()->data()};
        weights_writer->WriteTensor(out);
      }

      bool isAnchorStream = false;
      info.SetHandle(device.d2hId(tensor->id, isAnchorStream));
      builder.AddOutputModifiedParameter(device.h2dId(tensor->id), info);
    }
  }

  // Adding optimizers as inputs
  for (auto tensor : device.ir().optimizerTensors()) {
    ipu::TensorInfo info;
    info.SetHandle(device.h2dId(tensor->id));
    info.SetName(tensor->id);
    setIpuShape(info, tensor->info);
    builder.AddInput(info);
    if (weights_writer) {
      if (tensor->info.nbytes() != info.Shape().DataSizeInBytes()) {
        throw internal_error(
            "Size mismatch between metadata and data for optimizer {}",
            tensor->id);
      }
      info.SetType(ipu::TensorType::InputData);
      ipu::Tensor out{info, tensor->tensorData()->data()};
      weights_writer->WriteTensor(out);
    }
  }

  // List outfeeds
  for (TensorId anchorId : device.ir().getDataFlow().anchors()) {
    Tensor *tensor      = device.ir().getTensor(anchorId);
    bool isAnchorStream = true;
    ipu::TensorInfo info;
    info.SetHandle(device.d2hId(anchorId, isAnchorStream));
    info.SetName(tensor->id);
    setIpuShape(info, tensor->info);
    builder.CreateOutfeed(info.Name());
    builder.AddOutfeedStream(info.Name(), info);
  }

  // List feeds
  for (const auto &tensor : device.ir().dataStreamTensors()) {
    ipu::TensorInfo info;
    info.SetHandle(device.h2dId(tensor->id));
    info.SetName(tensor->id);
    setIpuShape(info, tensor->info);
    builder.CreateInfeed(info.Name());
    builder.AddInfeedStream(info.Name(), info);
  }
  return builder;
}
#endif // POPLAR_RUNNER
} // namespace

bool exporterIsAvailable() {
#ifndef POPLAR_RUNNER
  return false;
#else  // POPLAR_RUNNER
  return true;
#endif // POPLAR_RUNNER
}

void exportWeights(const Devicex &device, const std::string &weightsPath) {
#ifndef POPLAR_RUNNER
  throw(error(errorMsg, "export weights"));
#else  // POPLAR_RUNNER
  createBuilderAndExportWeights(device, &weightsPath);
#endif // POPLAR_RUNNER
}

void exportExecutable(poplar::Executable &executable,
                      const Devicex &device,
                      const poplar::OptionFlags &engineOptions,
                      const poplar::OptionFlags &deviceOptions,
                      const std::string &deviceHash,
                      int64_t numIPUs,
                      const std::string &executablePath) {
#ifndef POPLAR_RUNNER
  throw(error(errorMsg, "export an executable"));
#else  // POPLAR_RUNNER

  ipu::MetadataBuilder builder = createBuilderAndExportWeights(device);

  for (auto opt : engineOptions) {
    builder.AddEngineOption(opt.first, opt.second);
  }

  for (auto opt : deviceOptions) {
    builder.AddDeviceOption(opt.first, opt.second);
  }
  unsigned replication_factor = device.getReplicationFactor();
  builder.SetConfig(replication_factor, numIPUs);
  if (device.ir().getRequiresRandomSeed()) {
    builder.SetRandomNumberSeedHandle(
        device.h2dId(GetRandomSeedOp::getStreamedSeedTensorId()));
  }
  ipu::Metadata metadata = builder.BuildMetadata();

  std::string json_metadata = metadata.ToJson();
  // For security reasons don't store the verification information inside the
  // binary.
  metadata.verification_info.clear();
  ipu::BinaryWriter writer(executablePath + "/" + deviceHash + ".bin");
  writer.WriteMetadata(deviceHash, metadata);
  {
    ipu::ExecutableWriter execWriter = writer.CreateExecutable(deviceHash);
    executable.serialize(execWriter.Stream());
  }
  writer.Close();

  std::string metadataFilename = executablePath + "/" + deviceHash + ".json";
  std::ofstream metadata_file{metadataFilename};
  if (!metadata_file.is_open()) {
    throw error("Failed to open file {} for writing", metadataFilename);
  }
  metadata_file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  metadata_file.write(json_metadata.c_str(), json_metadata.size());
  metadata_file.close();
#endif // POPLAR_RUNNER
}

void exportStepIO(Builder &builder,
                  IStepIO &step,
                  int64_t numElements,
                  const std::vector<std::string> &feeds,
                  const std::string &outputFilename,
                  const std::string &metadataFilename) {
#ifndef POPLAR_RUNNER
  throw error(errorMsg, "export an IStepIO");
#else  // POPLAR_RUNNER
  ipu::BinaryWriter file(outputFilename);
  std::unique_ptr<ipu::Metadata> meta;
  if (!metadataFilename.empty()) {
    if (ipu::IsJsonFile(metadataFilename)) {
      meta = std::make_unique<ipu::Metadata>(
          ipu::LoadJsonFromFile(metadataFilename));
    } else {
      ipu::BinaryReader loader;
      loader.LoadFile(metadataFilename);
      meta = loader.ReadMetadata();
    }
  }
  for (const std::string &feed : feeds) {
    TensorInfo feedInfo{builder.getTensorDataType(feed),
                        builder.getTensorShape(feed)};
    ipu::TensorInfo ipuInfo;
    ipuInfo.SetName(feed);
    ipuInfo.SetType(ipu::TensorType::Infeed);
    setIpuShape(ipuInfo, feedInfo);

    ipu::FeedWriter writer = file.CreateFeed(feed, ipuInfo, numElements);
    if (meta) {
      validateInfeedInfo(*meta, feed, ipuInfo);
    }
    exportFeedContent(writer, feedInfo, numElements, step, feed);
  }
  file.Close();
#endif // POPLAR_RUNNER
}

void exportStepIO(IStepIO &step,
                  const Devicex &device,
                  int64_t numElements,
                  const std::string &outputFilename) {
#ifndef POPLAR_RUNNER
  throw error(errorMsg, "export an IStepIO");
#else  // POPLAR_RUNNER
  ipu::BinaryWriter file(outputFilename);
  for (Tensor *tensor : device.ir().dataStreamTensors()) {
    ipu::TensorInfo info;
    info.SetName(tensor->id);
    info.SetType(ipu::TensorType::Infeed);
    setIpuShape(info, tensor->info);
    std::vector<int32_t> conversionBuffer;
    ipu::FeedWriter writer = file.CreateFeed(info.Name(), info, numElements);
    exportFeedContent(writer, tensor->info, numElements, step, tensor->id);
  }
  file.Close();
#endif // POPLAR_RUNNER
}

} // namespace popx
} // namespace popart
