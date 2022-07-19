// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OPDEBUGINFO_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OPDEBUGINFO_HPP_

#include <popart/debugcontext.hpp>

namespace popart {
class Op;
class OpDebugInfo : public DebugInfo {
  const Op &op;
  bool finalizeCalled = false;

public:
  OpDebugInfo(const DebugContext &debugContext, const Op &_op);
  virtual ~OpDebugInfo();

  OpDebugInfo &operator=(const OpDebugInfo &) = delete;
  OpDebugInfo(const OpDebugInfo &)            = delete;

  // Called when the op is fully configured. Sets the debug info
  // for this op.
  void finalize();
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OPDEBUGINFO_HPP_
