// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUBGRAPHX_HPP
#define GUARD_NEURALNET_SUBGRAPHX_HPP

#include <popart/popx/popopx.hpp>
#include <popart/popx/preparedtensor.hpp>

#include "popart/names.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;

class SubgraphOpx : public PopOpx {
public:
  SubgraphOpx(Op *, Devicex *);
  bool outputCreatedExternally(OutIndex) const final { return true; }
  PreparedTensorInfos getInputsToPrepare() const override;
  PreparedTensorInfos getOutputsToPrepare() const override;

private:
};

} // namespace popx
} // namespace popart

#endif
