// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BUILDER_DEBUGINFO_HPP
#define GUARD_NEURALNET_BUILDER_DEBUGINFO_HPP

#include <map>
#include <string>
#include <vector>
#include <popart/debugcontext.hpp>
#include <poparttracepoint.hpp>

#include "popart/tensordebuginfo.hpp"

namespace popart {
class TensorInfo;
class any;

class BuilderDebugInfo : public DebugInfo {
public:
  // BuilderDebugInfo(const DebugContext &debugContext);

  BuilderDebugInfo(const DebugContext &debugContext,
                   const std::string &api,
                   const std::vector<TensorId> &inputs,
                   const std::map<std::string, popart::any> &attributes,
                   const std::vector<TensorId> &outputs = {});

  BuilderDebugInfo(const DebugContext &debugContext,
                   const std::string &api,
                   const std::map<std::string, popart::any> &args)
      : BuilderDebugInfo(debugContext, api, {}, args) {}

  BuilderDebugInfo(const DebugContext &debugContext,
                   const popart::string_view api,
                   const std::vector<TensorId> &inputs,
                   const std::map<std::string, popart::any> &attributes,
                   const std::vector<TensorId> &outputs = {})
      : BuilderDebugInfo(debugContext,
                         std::string(api.ptr, api.len),
                         inputs,
                         attributes,
                         outputs) {}

  void setOutputs(const std::vector<TensorId> &outputs);

  BuilderDebugInfo &operator=(const BuilderDebugInfo &) = delete;
  BuilderDebugInfo(const BuilderDebugInfo &)            = delete;
  virtual ~BuilderDebugInfo()                           = default;
};

class BuilderVarDebugInfo : public DebugInfo {
public:
  BuilderVarDebugInfo(const DebugContext &dc,
                      const std::string &api,
                      const TensorId &id,
                      const TensorInfo &ti);

  BuilderVarDebugInfo(const DebugContext &dc,
                      const std::string &api,
                      const TensorId &id);

  BuilderVarDebugInfo &operator=(const BuilderVarDebugInfo &) = delete;
  BuilderVarDebugInfo(const BuilderVarDebugInfo &)            = delete;
  virtual ~BuilderVarDebugInfo()                              = default;
};

} // namespace popart

#endif
