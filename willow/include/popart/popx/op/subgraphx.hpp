// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUBGRAPHX_HPP
#define GUARD_NEURALNET_SUBGRAPHX_HPP

#include <popart/popx/popopx.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {
namespace popx {

class SubgraphOpx : public PopOpx {
public:
  SubgraphOpx(Op *, Devicex *);
  bool outputCreatedExternally(OutIndex) const final { return true; }
  std::vector<std::tuple<TensorId, TensorId, bool>>
  getInputsToPrepare() const override;
  std::vector<std::tuple<TensorId, TensorId, bool>>
  getOutputsToPrepare() const override;

private:
};

} // namespace popx
} // namespace popart

#endif
