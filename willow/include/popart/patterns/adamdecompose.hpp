// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_ADAMDECOMPOSE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_ADAMDECOMPOSE_HPP_

#include <utility>
#include <vector>
#include <popart/patterns/optimizerdecompose.hpp>

#include "popart/tensordebuginfo.hpp"

namespace popart {
class AdamComboOp;
class Graph;
class Op;
class Tensor;

class AdamDecompose : public OptimizerDecompose {
public:
  bool matches(Op *) const final;
  std::vector<const Tensor *> touches(Op *) const final;
  bool apply(Op *) const final;

  TensorId rescaleRatio(Graph &graph, AdamComboOp *combo) const;
  std::pair<Op *, TensorId> rescaleAccl(Graph &graph,
                                        AdamComboOp *combo,
                                        bool accl1,
                                        TensorId acclId,
                                        TensorId gradIntoAcclId,
                                        TensorId rescaleRatioId) const;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_ADAMDECOMPOSE_HPP_
