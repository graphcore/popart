#include <fstream>

#include <boost/filesystem.hpp>

#include <popart/error.hpp>
#include <popart/filereader.hpp>
#include <popart/onnxutil.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {
namespace onnxutil {

// functions for translating between popart's enum class and onnx's enum
onnx::TensorProto_DataType getTPDataType(DataType data_type) {
  switch (data_type) {
  case DataType::UINT8: {
    return onnx::TensorProto_DataType_UINT8;
  }
  case DataType::INT8: {
    return onnx::TensorProto_DataType_INT8;
  }
  case DataType::UINT16: {
    return onnx::TensorProto_DataType_UINT16;
  }
  case DataType::INT16: {
    return onnx::TensorProto_DataType_INT16;
  }
  case DataType::INT32: {
    return onnx::TensorProto_DataType_INT32;
  }
  case DataType::INT64: {
    return onnx::TensorProto_DataType_INT64;
  }
  case DataType::UINT32: {
    return onnx::TensorProto_DataType_UINT32;
  }
  case DataType::UINT64: {
    return onnx::TensorProto_DataType_UINT64;
  }
  case DataType::BOOL: {
    return onnx::TensorProto_DataType_BOOL;
  }
  case DataType::FLOAT: {
    return onnx::TensorProto_DataType_FLOAT;
  }
  case DataType::FLOAT16: {
    return onnx::TensorProto_DataType_FLOAT16;
  }
  case DataType::BFLOAT16: {
    return onnx::TensorProto_DataType_BFLOAT16;
  }
  case DataType::DOUBLE: {
    return onnx::TensorProto_DataType_DOUBLE;
  }
  case DataType::COMPLEX64: {
    return onnx::TensorProto_DataType_COMPLEX64;
  }
  case DataType::COMPLEX128: {
    return onnx::TensorProto_DataType_COMPLEX128;
  }
  case DataType::STRING: {
    return onnx::TensorProto_DataType_STRING;
  }
  case DataType::UNDEFINED: {
    return onnx::TensorProto_DataType_UNDEFINED;
  }
  default:
    throw error("unrecognised PopART DataType");
  }
}

DataType getDataType(int tpd) {

  switch (tpd) {
  case onnx::TensorProto_DataType_UINT8: {
    return DataType::UINT8;
  }
  case onnx::TensorProto_DataType_INT8: {
    return DataType::INT8;
  }

  case onnx::TensorProto_DataType_UINT16: {
    return DataType::UINT16;
  }

  case onnx::TensorProto_DataType_INT16: {
    return DataType::INT16;
  }

  case onnx::TensorProto_DataType_INT32: {
    return DataType::INT32;
  }

  case onnx::TensorProto_DataType_INT64: {
    return DataType::INT64;
  }

  case onnx::TensorProto_DataType_UINT32: {
    return DataType::UINT32;
  }

  case onnx::TensorProto_DataType_UINT64: {
    return DataType::UINT64;
  }

  case onnx::TensorProto_DataType_BOOL: {
    return DataType::BOOL;
  }

  case onnx::TensorProto_DataType_FLOAT: {
    return DataType::FLOAT;
  }

  case onnx::TensorProto_DataType_FLOAT16: {
    return DataType::FLOAT16;
  }

  case onnx::TensorProto_DataType_BFLOAT16: {
    return DataType::BFLOAT16;
  }

  case onnx::TensorProto_DataType_DOUBLE: {
    return DataType::DOUBLE;
  }

  case onnx::TensorProto_DataType_COMPLEX64: {
    return DataType::COMPLEX64;
  }

  case onnx::TensorProto_DataType_COMPLEX128: {
    return DataType::COMPLEX128;
  }

  case onnx::TensorProto_DataType_STRING: {
    return DataType::STRING;
  }

  case onnx::TensorProto_DataType_UNDEFINED: {
    return DataType::UNDEFINED;
  }

  default:
    throw error("unrecognised ONNX DataType");
  }
}

ExternalTensorProtoInfo::ExternalTensorProtoInfo(const onnx::TensorProto &tp) {
  std::string name = tp.has_name() ? tp.name() : "";
  if (tp.has_data_location() &&
      tp.data_location() == onnx::TensorProto::EXTERNAL) {
    for (const auto &info : tp.external_data()) {
      if (info.key() == "location") {
        location = info.value();
      } else if (info.key() == "offset") {
        offset = std::stoi(info.value());
      } else if (info.key() == "length") {
        length = std::stoi(info.value());
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

ConstVoidData getConstData(const onnx::TensorProto &tp) {
  ConstVoidData cv_data;
  cv_data.info = TensorInfo(tp);
  cv_data.data = nullptr;
  // raw_data is a string: guaranteed to be contiguous in C++11
  if (tp.has_raw_data()) {
    cv_data.data = reinterpret_cast<const void *>(&(tp.raw_data()[0]));
  } else if (tp.has_data_location() &&
             tp.data_location() == onnx::TensorProto::EXTERNAL) {
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
    case onnx::TensorProto::COMPLEX64:
    case onnx::TensorProto::FLOAT: {
      cv_data.data = reinterpret_cast<const void *>(&(tp.float_data().Get(0)));
      break;
    }
    // onnx.proto states that COMPLEX128 is stored in the double field
    case onnx::TensorProto::COMPLEX128:
    case onnx::TensorProto::DOUBLE: {
      cv_data.data = reinterpret_cast<const void *>(&(tp.double_data().Get(0)));
      break;
    }
    case onnx::TensorProto::INT64: {
      cv_data.data = reinterpret_cast<const void *>(&(tp.int64_data().Get(0)));
      break;
    }

    // onnx.proto states that UINT32 is stored in the uint64 field
    case onnx::TensorProto::UINT64:
    case onnx::TensorProto::UINT32: {
      cv_data.data = reinterpret_cast<const void *>(&(tp.uint64_data().Get(0)));
      break;
    }

    // onnx.proto states that the following are stored as int32
    // field: INT32, INT16, INT8, UINT16, UINT8, BOOL, or FLOAT16
    case onnx::TensorProto::INT32:
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::UINT16:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::BOOL:
    case onnx::TensorProto::FLOAT16: {
      cv_data.data = reinterpret_cast<const void *>(&(tp.int32_data().Get(0)));
      break;
    }
    case onnx::TensorProto::UNDEFINED:
    case onnx::TensorProto::BFLOAT16:
    case onnx::TensorProto::STRING:
    default: {
      throw error("getConstData needs implementing for " +
                  cv_data.info.data_type());
    }
    }
  }

  return cv_data;
}

MutableVoidData getMutableData(onnx::TensorProto &tp) {
  MutableVoidData mv_data;
  mv_data.info = TensorInfo(tp);
  mv_data.data = nullptr;
  if (tp.has_raw_data()) {
    mv_data.data = reinterpret_cast<void *>(&((*tp.mutable_raw_data())[0]));
  } else if (tp.has_data_location() &&
             tp.data_location() == onnx::TensorProto::EXTERNAL) {
    throw error(
        "Cannot call getMutableData on tensor '{}', as it is stored externally",
        tp.name());
  } else {
    if (tp.data_type() == onnx::TensorProto::FLOAT) {
      mv_data.data =
          reinterpret_cast<void *>(tp.mutable_float_data()->Mutable(0));
    } else if (tp.data_type() == onnx::TensorProto::FLOAT16) {
      mv_data.data =
          reinterpret_cast<void *>(tp.mutable_int32_data()->Mutable(0));
    } else {
      throw error("getMutableData needs implementing for " +
                  mv_data.info.data_type());
    }
  }
  return mv_data;
}

onnx::ModelProto getModelProto(const std::string &modelProtoOrFilename) {
  onnx::ModelProto modelProto;
  if (io::isRegularFile(modelProtoOrFilename)) {
    modelProto = io::getModelFromFile(modelProtoOrFilename);
  } else {
    modelProto = io::getModelFromString(modelProtoOrFilename);
  }

  return modelProto;
}

namespace {
void clearInternallySavedData(onnx::TensorProto &tp) {
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

void saveInitializersExternally(onnx::ModelProto &model,
                                const std::vector<TensorId> &ids,
                                const std::string &fn) {

  if (boost::filesystem::exists(fn)) {
    logging::info("Saving tensor data to file {}, but it already exists. Its "
                  "contents will be overwritten",
                  fn);
  }
  std::ofstream ofs(fn, std::ofstream::binary);
  if (!ofs.is_open()) {
    throw error("Failed to open file {}", fn);
  }

  int64_t totalBytes = 0;
  for (const TensorId &id : ids) {
    // 1. Some checks:
    // That the tensor is an initializer
    if (!isInitializer(model, id)) {
      throw error("Tensor '{}' is not an initializer", id);
    }
    // That the tensor is not already saved externally
    onnx::TensorProto &tp = getTensorProto(model, id);
    if (tp.has_data_location()) {
      if (tp.data_location() == onnx::TensorProto::EXTERNAL) {
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
    tp.set_data_location(onnx::TensorProto::EXTERNAL);

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

onnx::TensorProto &getTensorProto(onnx::ModelProto &model,
                                  const TensorId &tId) {
  onnx::GraphProto *g = model.mutable_graph();

  for (unsigned i = 0; i < g->initializer_size(); ++i) {
    onnx::TensorProto &init = *g->mutable_initializer(i);
    if (init.name() == tId) {
      return init;
    }
  }
  throw error(
      "Could not find onnx::TensorProto with name {} in model initializer list",
      tId);
}

onnx::TensorProto getTensorProto(const onnx::ModelProto &model,
                                 const TensorId &tId) {
  onnx::GraphProto g = model.graph();

  for (unsigned i = 0; i < g.initializer_size(); ++i) {
    onnx::TensorProto &init = *g.mutable_initializer(i);
    if (init.name() == tId) {
      return init;
    }
  }
  throw error(
      "Could not find onnx::TensorProto with name {} in model initializer list",
      tId);
}

bool isInitializer(const onnx::ModelProto &model, const TensorId tId) {
  onnx::GraphProto g = model.graph();
  for (unsigned i = 0; i < g.initializer_size(); ++i) {
    onnx::TensorProto &init = *g.mutable_initializer(i);
    if (init.name() == tId) {
      return true;
    }
  }
  return false;
}

void visitModelNodes(onnx::ModelProto &model,
                     std::function<void(onnx::NodeProto &)> f) {
  onnx::GraphProto &g = *model.mutable_graph();

  for (unsigned node_i = 0; node_i < g.node_size(); ++node_i) {
    auto ptr_node         = g.mutable_node(node_i);
    onnx::NodeProto &node = *ptr_node;
    f(node);
  }
}

void visitModelInitializers(onnx::ModelProto &model,
                            std::function<void(onnx::TensorProto &)> f) {
  onnx::GraphProto &g = *model.mutable_graph();

  for (unsigned ii = 0; ii < g.initializer_size(); ++ii) {
    onnx::TensorProto &init = *g.mutable_initializer(ii);
    f(init);
  }
}

void visitModelValueInfos(onnx::ModelProto &model,
                          std::function<void(onnx::ValueInfoProto &)> f) {
  for (auto &vip : *model.mutable_graph()->mutable_output()) {
    f(vip);
  }
  for (auto &vip : *model.mutable_graph()->mutable_input()) {
    f(vip);
  }
}

} // namespace onnxutil
} // namespace popart
