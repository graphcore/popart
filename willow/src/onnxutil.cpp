// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <fstream>

#include <boost/filesystem.hpp>

#include <popart/error.hpp>
#include <popart/filereader.hpp>
#include <popart/onnxutil.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {
namespace onnxutil {

// functions for translating between popart's enum class and onnx's enum
ONNX_NAMESPACE::TensorProto_DataType getTPDataType(DataType data_type) {
  switch (data_type) {
  case DataType::UINT8: {
    return ONNX_NAMESPACE::TensorProto_DataType_UINT8;
  }
  case DataType::INT8: {
    return ONNX_NAMESPACE::TensorProto_DataType_INT8;
  }
  case DataType::UINT16: {
    return ONNX_NAMESPACE::TensorProto_DataType_UINT16;
  }
  case DataType::INT16: {
    return ONNX_NAMESPACE::TensorProto_DataType_INT16;
  }
  case DataType::INT32: {
    return ONNX_NAMESPACE::TensorProto_DataType_INT32;
  }
  case DataType::INT64: {
    return ONNX_NAMESPACE::TensorProto_DataType_INT64;
  }
  case DataType::UINT32: {
    return ONNX_NAMESPACE::TensorProto_DataType_UINT32;
  }
  case DataType::UINT64: {
    return ONNX_NAMESPACE::TensorProto_DataType_UINT64;
  }
  case DataType::BOOL: {
    return ONNX_NAMESPACE::TensorProto_DataType_BOOL;
  }
  case DataType::FLOAT: {
    return ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  }
  case DataType::FLOAT16: {
    return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  }
  case DataType::BFLOAT16: {
    return ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
  }
  case DataType::DOUBLE: {
    return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
  }
  case DataType::COMPLEX64: {
    return ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64;
  }
  case DataType::COMPLEX128: {
    return ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128;
  }
  case DataType::STRING: {
    return ONNX_NAMESPACE::TensorProto_DataType_STRING;
  }
  case DataType::UNDEFINED: {
    return ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  }
  default:
    throw error("unrecognised PopART DataType");
  }
}

DataType getDataType(int tpd) {

  switch (tpd) {
  case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
    return DataType::UINT8;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
    return DataType::INT8;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
    return DataType::UINT16;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
    return DataType::INT16;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
    return DataType::INT32;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
    return DataType::INT64;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_UINT32: {
    return DataType::UINT32;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
    return DataType::UINT64;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_BOOL: {
    return DataType::BOOL;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
    return DataType::FLOAT;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
    return DataType::FLOAT16;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
    return DataType::BFLOAT16;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
    return DataType::DOUBLE;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
    return DataType::COMPLEX64;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {
    return DataType::COMPLEX128;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_STRING: {
    return DataType::STRING;
  }

  case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED: {
    return DataType::UNDEFINED;
  }

  default:
    throw error("unrecognised ONNX DataType");
  }
}

ExternalTensorProtoInfo::ExternalTensorProtoInfo(
    const ONNX_NAMESPACE::TensorProto &tp) {
  std::string name = tp.has_name() ? tp.name() : "";
  if (tp.has_data_location() &&
      tp.data_location() == ONNX_NAMESPACE::TensorProto::EXTERNAL) {
    for (const auto &info : tp.external_data()) {
      if (info.key() == "location") {
        location = info.value();
      } else if (info.key() == "offset") {
        offset = std::stol(info.value());
      } else if (info.key() == "length") {
        length = std::stol(info.value());
      }
    }

    // Check the external_data contains valid settings
    if (location == "") {
      throw error("No location information for externally stored tensor '{}'",
                  name);
    } else if (!boost::filesystem::exists(location)) {
      throw error(
          "Unrecognised file name '{}' for externally stored tensor '{}'",
          location,
          name);
    }
    if (length <= 0) {
      throw error(
          "Invalid 'length' information ({}) for externally stored tensor '{}'",
          length,
          name);
    }
    if (offset < 0) {
      throw error(
          "Invalid 'offset' information ({}) for externally stored tensor '{}'",
          offset,
          name);
    }
  } else {
    throw error("Cannot determine ExternalTensorProtoInfo for '{}', as it does "
                "not have an external data location",
                name);
  }
}

ConstVoidData getConstData(const ONNX_NAMESPACE::TensorProto &tp) {
  ConstVoidData cv_data;
  cv_data.info = TensorInfo(tp);
  cv_data.data = nullptr;

  // It is possible to add an onnx Constant op with no elements:
  //   builder.aiOnnxOpset11.constant(np.array([], dtype=np.float32), False)
  if (cv_data.info.nelms() == 0) {
    cv_data.data = nullptr;
  }
  // raw_data is a string: guaranteed to be contiguous in C++11
  else if (tp.has_raw_data()) {
    cv_data.data = reinterpret_cast<const void *>(&(tp.raw_data()[0]));
  } else if (tp.has_data_location() &&
             tp.data_location() == ONNX_NAMESPACE::TensorProto::EXTERNAL) {
    auto externalInfo = ExternalTensorProtoInfo(tp);

    std::ifstream ifs(externalInfo.location, std::ios::binary);
    if (!ifs.is_open()) {
      throw error("Unable to open file '{}' to read tensor data for '{}'",
                  externalInfo.location,
                  tp.name());
    }

    if (externalInfo.offset > 0) {
      ifs.seekg(externalInfo.offset, std::ios::beg);
    }
    std::vector<char> externalTensorBuffer(externalInfo.length);
    ifs.read(externalTensorBuffer.data(), externalInfo.length);
    cv_data.store(std::move(externalTensorBuffer), TensorInfo(tp));
    ifs.close();
  }
  // note: protobuf repeated field is essentially stl vector (has a function
  // Capactity()) and so data is contiguous
  else {
    switch (tp.data_type()) {
    // onnx.proto states that COMPLEX64 is stored in the float field
    case ONNX_NAMESPACE::TensorProto::COMPLEX64:
    case ONNX_NAMESPACE::TensorProto::FLOAT: {
      cv_data.data = reinterpret_cast<const void *>(&(tp.float_data().Get(0)));
      break;
    }
    // onnx.proto states that COMPLEX128 is stored in the double field
    case ONNX_NAMESPACE::TensorProto::COMPLEX128:
    case ONNX_NAMESPACE::TensorProto::DOUBLE: {
      cv_data.data = reinterpret_cast<const void *>(&(tp.double_data().Get(0)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto::INT64: {
      cv_data.data = reinterpret_cast<const void *>(&(tp.int64_data().Get(0)));
      break;
    }

    // onnx.proto states that UINT32 is stored in the uint64 field
    case ONNX_NAMESPACE::TensorProto::UINT64:
    case ONNX_NAMESPACE::TensorProto::UINT32: {
      cv_data.data = reinterpret_cast<const void *>(&(tp.uint64_data().Get(0)));
      break;
    }

    // onnx.proto states that the following are stored as int32
    // field: INT32, INT16, INT8, UINT16, UINT8, BOOL, or FLOAT16
    case ONNX_NAMESPACE::TensorProto::INT32:
    case ONNX_NAMESPACE::TensorProto::INT16:
    case ONNX_NAMESPACE::TensorProto::INT8:
    case ONNX_NAMESPACE::TensorProto::UINT16:
    case ONNX_NAMESPACE::TensorProto::UINT8:
    case ONNX_NAMESPACE::TensorProto::BOOL:
    case ONNX_NAMESPACE::TensorProto::FLOAT16: {
      cv_data.data = reinterpret_cast<const void *>(&(tp.int32_data().Get(0)));
      break;
    }
    case ONNX_NAMESPACE::TensorProto::UNDEFINED:
    case ONNX_NAMESPACE::TensorProto::BFLOAT16:
    case ONNX_NAMESPACE::TensorProto::STRING:
    default: {
      throw error("getConstData needs implementing for " +
                  cv_data.info.data_type());
    }
    }
  }

  return cv_data;
}

MutableVoidData getMutableData(ONNX_NAMESPACE::TensorProto &tp) {
  MutableVoidData mv_data;
  mv_data.info = TensorInfo(tp);
  mv_data.data = nullptr;
  if (tp.has_raw_data()) {
    mv_data.data = reinterpret_cast<void *>(&((*tp.mutable_raw_data())[0]));
  } else if (tp.has_data_location() &&
             tp.data_location() == ONNX_NAMESPACE::TensorProto::EXTERNAL) {
    throw error(
        "Cannot call getMutableData on tensor '{}', as it is stored externally",
        tp.name());
  } else {
    if (tp.data_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
      mv_data.data =
          reinterpret_cast<void *>(tp.mutable_float_data()->Mutable(0));
    } else if (tp.data_type() == ONNX_NAMESPACE::TensorProto::FLOAT16) {
      mv_data.data =
          reinterpret_cast<void *>(tp.mutable_int32_data()->Mutable(0));
    } else if (tp.data_type() == ONNX_NAMESPACE::TensorProto::DOUBLE) {
      mv_data.data =
          reinterpret_cast<void *>(tp.mutable_double_data()->Mutable(0));
    } else if (tp.data_type() == ONNX_NAMESPACE::TensorProto::INT64) {
      mv_data.data =
          reinterpret_cast<void *>(tp.mutable_int64_data()->Mutable(0));
    } else {
      throw error("getMutableData needs implementing for " +
                  mv_data.info.data_type());
    }
  }
  return mv_data;
}

ONNX_NAMESPACE::ModelProto
getModelProto(const std::string &modelProtoOrFilename) {
  ONNX_NAMESPACE::ModelProto modelProto;
  if (io::isRegularFile(modelProtoOrFilename)) {
    modelProto = io::getModelFromFile(modelProtoOrFilename);
  } else {
    modelProto = io::getModelFromString(modelProtoOrFilename);
  }

  return modelProto;
}

namespace {
void clearInternallySavedData(ONNX_NAMESPACE::TensorProto &tp) {
  if (tp.has_raw_data()) {
    tp.clear_raw_data();
  }

  // Determine all types of data that can be stored in a TensorProto with:
  // $ grep "_data_.Clear(" onnx/onnx.pb.h
  tp.clear_float_data();  // float_data_.Clear();
  tp.clear_int32_data();  // int32_data_.Clear();
  tp.clear_string_data(); // string_data_.Clear();
  tp.clear_int64_data();  // int64_data_.Clear();
                          // external_data_.Clear();
  tp.clear_double_data(); // double_data_.Clear();
  tp.clear_uint64_data(); // uint64_data_.Clear();
}
} // namespace

bool isExternallySavedInitializer(ONNX_NAMESPACE::ModelProto &model,
                                  const TensorId &id) {
  if (!isInitializer(model, id)) {
    throw error("Tensor '{}' is not an initializer", id);
  }

  ONNX_NAMESPACE::TensorProto &tp = getTensorProto(model, id);
  if (tp.has_data_location()) {
    if (tp.data_location() == ONNX_NAMESPACE::TensorProto::EXTERNAL) {
      return true;
    }
  }
  return false;
}

std::string getExternallySavedTensorLocation(ONNX_NAMESPACE::ModelProto &model,
                                             const TensorId &id) {
  if (!isExternallySavedInitializer(model, id)) {
    throw error("Tensor '{}' is not an externally saved initializer", id);
  }

  ONNX_NAMESPACE::TensorProto &tp = getTensorProto(model, id);
  return ExternalTensorProtoInfo(tp).location;
}

void saveInitializersExternally(ONNX_NAMESPACE::ModelProto &model,
                                const std::vector<TensorId> &ids,
                                const std::string &fn,
                                bool appendToExistingFile) {

  if (boost::filesystem::exists(fn)) {
    if (!appendToExistingFile) {
      logging::info("Saving tensor data to file {}, but it already exists. Its "
                    "contents will be overwritten",
                    fn);
    }
  } else {
    if (appendToExistingFile) {
      std::string idsString;
      for (const auto &s : ids) {
        idsString = idsString + s + " ";
      }
      throw error("TensorIds '{}' to be saved externally to existing file {}, "
                  "but file doesn't exist",
                  idsString,
                  fn);
    }
  }

  int64_t totalBytes;
  if (appendToExistingFile) {
    totalBytes = static_cast<int64_t>(boost::filesystem::file_size(fn));
  } else {
    totalBytes = 0;
  }

  std::ofstream ofs;
  if (appendToExistingFile) {
    ofs.open(fn, std::ofstream::binary | std::ofstream::app);
  } else {
    ofs.open(fn, std::ofstream::binary);
  }
  if (!ofs.is_open()) {
    throw error("Failed to open file {}", fn);
  }

  for (const TensorId &id : ids) {
    // 1. Some checks:
    // That the tensor is an initializer
    if (!isInitializer(model, id)) {
      throw error("Tensor '{}' is not an initializer", id);
    }
    // That the tensor is not already saved externally
    ONNX_NAMESPACE::TensorProto &tp = getTensorProto(model, id);
    if (tp.has_data_location()) {
      if (tp.data_location() == ONNX_NAMESPACE::TensorProto::EXTERNAL) {
        throw error("Tensor '{}' already has an external data_location", id);
      }
    }
    // And that the tensor has data that can be saved externally
    ConstVoidData cvData = getConstData(tp);
    auto nBytes          = cvData.info.nbytes();
    if (nBytes == 0) {
      throw error("Tensor '{}' has no data to save externally", id);
    }

    // 2. Set data location to external
    tp.set_data_location(ONNX_NAMESPACE::TensorProto::EXTERNAL);

    // 3. Set external_data field
    auto externalDataInfo = tp.mutable_external_data();
    auto *location        = externalDataInfo->Add();
    location->set_key("location");
    location->set_value(fn);

    auto *length = externalDataInfo->Add();
    length->set_key("length");
    length->set_value(std::to_string(nBytes));

    auto *offset = externalDataInfo->Add();
    offset->set_key("offset");
    offset->set_value(std::to_string(totalBytes));
    totalBytes += nBytes;

    // 4. Save data to file
    ofs.write(static_cast<const char *>(cvData.data), nBytes);

    // 5. Delete the internally saved tensor data
    clearInternallySavedData(tp);
  }

  ofs.close();
}

ONNX_NAMESPACE::TensorProto &getTensorProto(ONNX_NAMESPACE::ModelProto &model,
                                            const TensorId &tId) {
  ONNX_NAMESPACE::GraphProto *g = model.mutable_graph();

  for (unsigned i = 0; i < g->initializer_size(); ++i) {
    ONNX_NAMESPACE::TensorProto &init = *g->mutable_initializer(i);
    if (init.name() == tId) {
      return init;
    }
  }
  throw error("Could not find ONNX_NAMESPACE::TensorProto with name {} in "
              "model initializer list",
              tId);
}

ONNX_NAMESPACE::TensorProto
getTensorProto(const ONNX_NAMESPACE::ModelProto &model, const TensorId &tId) {
  ONNX_NAMESPACE::GraphProto g = model.graph();

  for (unsigned i = 0; i < g.initializer_size(); ++i) {
    ONNX_NAMESPACE::TensorProto &init = *g.mutable_initializer(i);
    if (init.name() == tId) {
      return init;
    }
  }
  throw error("Could not find ONNX_NAMESPACE::TensorProto with name {} in "
              "model initializer list",
              tId);
}

bool isInitializer(const ONNX_NAMESPACE::ModelProto &model,
                   const TensorId tId) {
  ONNX_NAMESPACE::GraphProto g = model.graph();
  for (unsigned i = 0; i < g.initializer_size(); ++i) {
    ONNX_NAMESPACE::TensorProto &init = *g.mutable_initializer(i);
    if (init.name() == tId) {
      return true;
    }
  }
  return false;
}

void visitModelNodes(ONNX_NAMESPACE::ModelProto &model,
                     std::function<void(ONNX_NAMESPACE::NodeProto &)> f) {
  ONNX_NAMESPACE::GraphProto &g = *model.mutable_graph();

  for (unsigned node_i = 0; node_i < g.node_size(); ++node_i) {
    auto ptr_node                   = g.mutable_node(node_i);
    ONNX_NAMESPACE::NodeProto &node = *ptr_node;
    f(node);
  }
}

void visitModelInitializers(
    ONNX_NAMESPACE::ModelProto &model,
    std::function<void(ONNX_NAMESPACE::TensorProto &)> f) {
  ONNX_NAMESPACE::GraphProto &g = *model.mutable_graph();

  for (unsigned ii = 0; ii < g.initializer_size(); ++ii) {
    ONNX_NAMESPACE::TensorProto &init = *g.mutable_initializer(ii);
    f(init);
  }
}

void visitModelValueInfos(
    ONNX_NAMESPACE::ModelProto &model,
    std::function<void(ONNX_NAMESPACE::ValueInfoProto &)> f) {
  for (auto &vip : *model.mutable_graph()->mutable_output()) {
    f(vip);
  }
  for (auto &vip : *model.mutable_graph()->mutable_input()) {
    f(vip);
  }
}

} // namespace onnxutil
} // namespace popart
