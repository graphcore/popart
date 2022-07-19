// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_ONNXDEBUGINFO_HPP_
#define POPART_WILLOW_INCLUDE_POPART_ONNXDEBUGINFO_HPP_

#include <popart/debugcontext.hpp>
#include <popart/names.hpp>

namespace onnx {
class TensorProto;
class ValueInfoProto;
} // namespace onnx

namespace popart {
class TensorInfo;

class OnnxOpDebugInfo : public DebugInfo {
public:
  OnnxOpDebugInfo(const DebugContext &debugContext, const Node &node);
  OnnxOpDebugInfo &operator=(const OnnxOpDebugInfo &) = delete;
  OnnxOpDebugInfo(const OnnxOpDebugInfo &)            = delete;
  virtual ~OnnxOpDebugInfo()                          = default;
};

class OnnxVariableDebugInfo : public DebugInfo {

public:
  OnnxVariableDebugInfo(const DebugContext &dc,
                        const ONNX_NAMESPACE::TensorProto &proto);

  OnnxVariableDebugInfo(const DebugContext &dc,
                        const ONNX_NAMESPACE::ValueInfoProto &proto);

  OnnxVariableDebugInfo(const DebugContext &dc,
                        const ONNX_NAMESPACE::ValueInfoProto &proto,
                        const TensorInfo &ti);

  OnnxVariableDebugInfo &operator=(const OnnxVariableDebugInfo &) = delete;
  OnnxVariableDebugInfo(const OnnxVariableDebugInfo &)            = delete;
  virtual ~OnnxVariableDebugInfo()                                = default;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_ONNXDEBUGINFO_HPP_
