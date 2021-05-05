// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD2DECOMPOSE_PATTERN_HPP
#define GUARD_NEURALNET_SGD2DECOMPOSE_PATTERN_HPP

#include <popart/patterns/optimizerdecompose.hpp>
#include <popart/patterns/patterns.hpp>

namespace popart {

class Graph;
class Op;
class SGD2ComboOp;

class SGD2Decompose : public OptimizerDecompose {
public:
  bool matches(Op *) const final;
  std::vector<const Tensor *> touches(Op *) const final;
  bool apply(Op *) const final;

private:
  TensorId acclUpdate(Graph &graph,
                      const SGD2ComboOp *combo,
                      const TensorId &gradIntoAcclId,
                      const TensorId &accl1Id,
                      const TensorId &weightId) const;

  void varUpdateAndEraseCombo(Graph &graph,
                              SGD2ComboOp *combo,
                              const TensorId &weightId,
                              const TensorId &updatedAcc1lId,
                              const TensorId &updatedWeightId) const;
};

} // namespace popart

#endif