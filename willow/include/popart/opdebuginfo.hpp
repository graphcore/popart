// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OP_DEBUGINFO_HPP
#define GUARD_NEURALNET_OP_DEBUGINFO_HPP

#include <popart/debugcontext.hpp>

namespace popart {
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

#endif