// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iosfwd>
#include <onnx/onnx_pb.h>
#include <string>
#include <vector>
#include <popart/onnxdebuginfo.hpp>
#include <popart/util.hpp>

#include "popart/debugcontext.hpp"
#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"

namespace {
using namespace popart;

ProfileValue toProfileValue(const ONNX_NAMESPACE::TensorProto &proto) {
  ProfileValue::Map tensorProto;
  tensorProto.insert({"name", proto.name()});

  TensorInfo ti(proto);
  std::stringstream ss;
  ss << ti.shape();

  tensorProto.insert({"dims", ss.str()});

  tensorProto.insert({"dataType",
                      ONNX_NAMESPACE::TensorProto::DataType_Name(
                          static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
                              proto.data_type()))});

  return tensorProto;
}

ProfileValue toProfileValue(const ONNX_NAMESPACE::ValueInfoProto &proto) {
  ProfileValue::Map valueInfoProto;
  valueInfoProto.insert({"name", proto.name()});

  if (proto.has_type()) {
    ProfileValue::Map typeProto;
    if (proto.type().tensor_type().has_shape()) {

      TensorInfo ti(proto.type());
      std::stringstream ss;
      ss << ti.shape();

      typeProto.insert({"shape", ss.str()});
    }
    valueInfoProto.insert({"type", typeProto});
  }
  return valueInfoProto;
}
} // namespace

namespace popart {

OnnxOpDebugInfo::OnnxOpDebugInfo(const DebugContext &debugContext,
                                 const Node &node)
    : DebugInfo(debugContext, "onnx") {

  setValue("category", ProfileValue{"op"});

  ProfileValue::Vector inputs;
  for (int i = 0; i < node.input_size(); i++) {
    inputs.push_back(node.input(i));
  }
  setValue("input", inputs);

  ProfileValue::Vector outputs;
  for (int i = 0; i < node.output_size(); i++) {
    outputs.push_back(node.output(i));
  }
  setValue("output", outputs);
  setValue("opName", node.name());
  setValue("opType", node.op_type());
  setValue("domain", node.domain());

  ProfileValue::Map attributes;
  for (auto &attr : node.attribute()) {

    ProfileValue value = "";

    switch (attr.type()) {
    case ONNX_NAMESPACE::AttributeProto::UNDEFINED:
      attributes.insert({attr.name(), "UNDEFINED"});
      break;
    case ONNX_NAMESPACE::AttributeProto::FLOAT:
      attributes.insert({attr.name(), attr.f()});
      break;
    case ONNX_NAMESPACE::AttributeProto::INT:
      attributes.insert({attr.name(), attr.i()});
      break;
    case ONNX_NAMESPACE::AttributeProto::STRING:
      attributes.insert({attr.name(), attr.s()});
      break;
    case ONNX_NAMESPACE::AttributeProto::TENSOR: {
      attributes.insert({attr.name(), toProfileValue(attr.t())});
    } break;
    case ONNX_NAMESPACE::AttributeProto::INTS: {
      std::stringstream ss;
      appendSequence(ss, attr.ints());
      attributes.insert({attr.name(), ss.str()});
    } break;
    case ONNX_NAMESPACE::AttributeProto::FLOATS: {
      std::stringstream ss;
      appendSequence(ss, attr.floats());
      attributes.insert({attr.name(), ss.str()});
    } break;
    case ONNX_NAMESPACE::AttributeProto::STRINGS: {
      std::stringstream ss;
      appendSequence(ss, attr.ints());
      attributes.insert({attr.name(), ss.str()});
    } break;
    case ONNX_NAMESPACE::AttributeProto::GRAPH:
    case ONNX_NAMESPACE::AttributeProto::TENSORS:
    case ONNX_NAMESPACE::AttributeProto::GRAPHS:
    case ONNX_NAMESPACE::AttributeProto::SPARSE_TENSOR:
    case ONNX_NAMESPACE::AttributeProto::SPARSE_TENSORS:
    default:
      attributes.insert({attr.name(), "<TYPE NOT SUPPORTED>"});
      break;
    };
  }

  setValue("attribute", attributes);
}

OnnxVariableDebugInfo::OnnxVariableDebugInfo(
    const DebugContext &debugContext,
    const ONNX_NAMESPACE::TensorProto &proto)
    : DebugInfo(debugContext, "onnx") {
  setValue("category", ProfileValue{"variable"});
  setValue("tensorProto", toProfileValue(proto));
}

OnnxVariableDebugInfo::OnnxVariableDebugInfo(
    const DebugContext &debugContext,
    const ONNX_NAMESPACE::ValueInfoProto &proto)
    : DebugInfo(debugContext, "onnx") {
  setValue("category", ProfileValue{"variable"});
  setValue("valueInfoProto", toProfileValue(proto));
}

OnnxVariableDebugInfo::OnnxVariableDebugInfo(
    const DebugContext &debugContext,
    const ONNX_NAMESPACE::ValueInfoProto &proto,
    const TensorInfo &ti)
    : DebugInfo(debugContext, "onnx") {
  setValue("category", ProfileValue{"variable"});

  ProfileValue::Map valueInfoProto;
  valueInfoProto.insert({"name", proto.name()});

  ProfileValue::Map typeProto;
  std::stringstream ss;
  ss << ti.shape();
  valueInfoProto.insert({"shape_from_input", ss.str()});

  setValue("valueInfoProto", valueInfoProto);
}

} // namespace popart
