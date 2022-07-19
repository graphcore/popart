// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TENSORDEBUGINFO_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TENSORDEBUGINFO_HPP_

#include <string>
#include <popart/debugcontext.hpp>

namespace popart {

using TensorId = std::string;
class TensorInfo;

enum class TensorType;

class TensorDebugInfo : public DebugInfo {
public:
  TensorDebugInfo(const DebugContext &debugContext,
                  const TensorId &tenid,
                  const TensorInfo &info,
                  const TensorType &tt);

  TensorDebugInfo(const DebugContext &debugContext,
                  const TensorId &tenid,
                  const TensorType &tt);

  TensorDebugInfo &operator=(const TensorDebugInfo &) = delete;
  TensorDebugInfo(const TensorDebugInfo &)            = delete;
  virtual ~TensorDebugInfo()                          = default;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TENSORDEBUGINFO_HPP_
