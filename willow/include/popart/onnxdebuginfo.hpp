// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNX_DEBUGINFO_HPP
#define GUARD_NEURALNET_ONNX_DEBUGINFO_HPP

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

#endif
