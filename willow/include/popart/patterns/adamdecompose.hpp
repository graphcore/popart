// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADAMDECOMPOSE_PATTERN_HPP
#define GUARD_NEURALNET_ADAMDECOMPOSE_PATTERN_HPP

#include <popart/op/adamcombo.hpp>
#include <popart/patterns/optimizerdecompose.hpp>
#include <popart/patterns/patterns.hpp>

namespace popart {

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

#endif
