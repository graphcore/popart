// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUBGRAPHX_HPP
#define GUARD_NEURALNET_SUBGRAPHX_HPP

#include <popart/popx/opx.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {
namespace popx {

class SubgraphOpx : public Opx {
public:
  SubgraphOpx(Op *, Devicex *);
  bool outputCreatedExternally(OutIndex) const final { return true; }
  std::vector<std::tuple<TensorId, TensorId, bool>>
  getOutputsToPrepare() const override;

private:
};

} // namespace popx
} // namespace popart

#endif
