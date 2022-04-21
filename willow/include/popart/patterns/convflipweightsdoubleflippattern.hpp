// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONVFLIPWEIGHTS_DOUBLE_FLIP_PATTERN_HPP
#define GUARD_NEURALNET_CONVFLIPWEIGHTS_DOUBLE_FLIP_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

// If two ConvFlipWeightsOps are connected to each other, they cancel each
// other out and can be bypassed with the second op removed
//
//
// Example use case
//
// Before this pattern
//
//   o_grad ---------------------------------.
//      |                                    |
// w ---+-------> ConvFlipWeight---------.   |
//      |              |                 |   |
//      |        ConvFlipWeights   ConvWeightsGrad
//      |              |                   |
//      '----------> Conv          ConvFlipWeights
//                     |                   |
//                   d_grad              w_grad
//
// After this pattern
//
//   o_grad --------------------------------------------.
//      |                                               |
// w ---+------------------> ConvFlipWeight---------.   |
//      |              |                            |   |
//      '----------> Conv                     ConvWeightsGrad
//                     |                              |
//                   d_grad                   ConvFlipWeights
//                                                    |
//                                                  w_grad

class ConvFlipWeightsDoubleFlipPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif
