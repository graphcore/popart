// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_CONVFLIPWEIGHTSDOUBLEFLIPPATTERN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_CONVFLIPWEIGHTSDOUBLEFLIPPATTERN_HPP_

#include <vector>

#include "popart/patterns/pattern.hpp"

namespace popart {
class Op;
class Tensor;

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

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_CONVFLIPWEIGHTSDOUBLEFLIPPATTERN_HPP_
