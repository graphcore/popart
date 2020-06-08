// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUBGRAPHX_HPP
#define GUARD_NEURALNET_SUBGRAPHX_HPP

#include <boost/optional.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class SubgraphOpx : public Opx {
public:
  SubgraphOpx(Op *, Devicex *);

private:
};

} // namespace popx
} // namespace popart

#endif
