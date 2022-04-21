// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONVFLIPWEIGHT_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_CONVFLIPWEIGHT_GRAD_OP_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

// These ASCII diagrams show the steps before this pattern as well as the effect
// of this pattern.
//
// NB lower case means a tensor, uppercase, the Op name
// + means a crossing without a junction i.e. which is *not* connected between
// the horizontal and vertical
//
// After ConvTransposePattern
//
// w --> ConvFlipWeights
//               |
// d --------> Conv --> o
//
//
// After Autodiff
//
//   o_grad ------------------------------.
//      |                                 |
// w ---+----> ConvFlipWeights -------.   |
//      |              |              |   |
//      '-------> ConvDataGrad   ConvWeightsGrad
//                     |                |
//                   d_grad   ConvFlipWeightsGradOp
//                                      |
//                                     w_grad
//
//
// After ConvDataGradPattern
//
//   o_grad ---------------------------------.
//      |                                    |
// w ---+-------> ConvFlipWeight---------.   |
//      |              |                 |   |
//      |        ConvFlipWeights   ConvWeightsGrad
//      |              |                   |
//      '----------> Conv       ConvFlipWeightsGrad
//                     |                   |
//                   d_grad              w_grad
//
//
// After this pattern
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
// (An additional pattern can remove the double flip by short circuiting the
// weight in to the Conv and removing the second flip)

// Replace a ConvFlipWeightsGradOp with ConvFlipWeightsOp
class ConvFlipWeightsGradOpPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif
