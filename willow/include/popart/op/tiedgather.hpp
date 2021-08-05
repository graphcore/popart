// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef GUARD_NEURALNET_TIEDGATHER_HPP
#define GUARD_NEURALNET_TIEDGATHER_HPP

#include <popart/op/gather.hpp>

namespace popart {

class TiedGatherOp final : public GatherOp {
public:
  TiedGatherOp(int64_t axis_, const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  bool checkIndices() const { return checkIndices_; }

private:
  bool checkIndices_ = true;
};

class TiedGatherGradOp final : public GatherGradOp {
public:
  TiedGatherGradOp(const TiedGatherOp *fwdOp, int64_t axis);

  std::unique_ptr<Op> clone() const final;

  const TiedGatherOp *fwdOp;
};

} // namespace popart

#endif
