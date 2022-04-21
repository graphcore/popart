// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/tensor.hpp>
#include <popart/tensordebuginfo.hpp>

namespace {
using namespace popart;
std::string to_string(const TensorType &tt) {
  std::stringstream ss;
  ss << tt;
  return ss.str();
}

std::string to_string(const DataType &dt) {
  std::stringstream ss;
  ss << dt;
  return ss.str();
}

std::string to_string(const Shape &shape) {
  std::stringstream ss;
  ss << shape;
  return ss.str();
}
} // namespace

namespace popart {

TensorDebugInfo::TensorDebugInfo(const DebugContext &debugContext,
                                 const TensorId &tenid,
                                 const TensorInfo &info,
                                 const TensorType &tt)
    : DebugInfo(debugContext, "popart") {
  setValue("category", ProfileValue{"tensor"});
  setValue("tensorId", ProfileValue{tenid});
  setValue("shape", ProfileValue{to_string(info.shape())});
  if (info.getDataTypeInfo() != nullptr) {
    setValue("elementType", ProfileValue{to_string(info.dataType())});
  }
  setValue("type", ProfileValue{to_string(tt)});
}

TensorDebugInfo::TensorDebugInfo(const DebugContext &debugContext,
                                 const TensorId &tenid,
                                 const TensorType &tt)
    : DebugInfo(debugContext, "popart") {
  setValue("category", ProfileValue{"tensor"});
  setValue("tensorId", ProfileValue{tenid});
  setValue("type", ProfileValue{to_string(tt)});
}
} // namespace popart