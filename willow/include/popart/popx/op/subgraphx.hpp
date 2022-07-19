// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SUBGRAPHX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SUBGRAPHX_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SUBGRAPHX_HPP_
