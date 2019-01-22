#include <poponnx/error.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {
namespace onnxutil {

// functions for translating between poponnx's enum class and onnx's enum
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
    throw error("unrecognised PopONNX DataType");
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

ConstVoidData getConstData(const onnx::TensorProto &tp) {
  ConstVoidData cv_data;
  cv_data.info = TensorInfo(tp);
  cv_data.data = nullptr;
  // raw_data is a string: guaranteed to be contiguous in C++11
  if (tp.has_raw_data()) {
    cv_data.data = reinterpret_cast<const void *>(&(tp.raw_data()[0]));
  }
  // note: protobuf repeated field is essentially stl vector (has a function
  // Capactity()) and so data is contiguous
  else {
    switch (tp.data_type()) {
    // we glean from onnx.proto that COMPLEX64 is stored in the float field
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
} // namespace poponnx
